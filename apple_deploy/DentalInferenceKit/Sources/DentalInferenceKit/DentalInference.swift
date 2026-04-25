/// On-device dental patch classification for iOS, macOS, and visionOS.
///
/// Setup:
///   1. Run `export_coreml.py` → `dental_patch_encoder.mlpackage`
///   2. Run `save_text_emb_bin.py` → `text_embeddings.bin`
///   3. Place both in DentalInferenceKit/Sources/DentalInferenceKit/Resources/
///
/// Usage:
///   let results = try DentalInference.shared.predict(xyz: flatXYZ)
import CoreML
import Accelerate
import Foundation

// MARK: - Result types

public struct PatchPrediction: Sendable {
    public let patchIndex: Int
    public let label: String
    public let score: Float
    public let centerXYZ: (x: Float, y: Float, z: Float)
}

public struct PatchEmbeddingOutput: Sendable {
    /// Number of patch groups (G)
    public let groupCount: Int
    /// (G * 3) normalised patch centre coordinates, row-major
    public let centers: [Float]
    /// (G * 3) original-scale patch centre coordinates, row-major
    public let denormalizedCenters: [Float]
    /// (G * embeddingDim) patch embeddings, row-major
    public let embeddings: [Float]
    public let embeddingDim: Int
}

// MARK: - Main class

public final class DentalInference: @unchecked Sendable {

    public static let shared = DentalInference()

    public var numGroup:  Int = 128
    public var groupSize: Int = 32
    public var clipDim:   Int = 1280

    private let model:   MLModel
    // Loaded lazily on first predict() call; nil if text_embeddings.bin is absent.
    private let textEmb: [Float]?
    private let labels:  [String]?

    // MARK: - Init

    private init() {
        let bundle = Bundle.module

        guard let modelURL = bundle.url(
            forResource: "dental_patch_encoder", withExtension: "mlmodelc"
        ) else { fatalError("dental_patch_encoder.mlmodelc not found in bundle") }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all   // Neural Engine + GPU where available
        guard let m = try? MLModel(contentsOf: modelURL, configuration: cfg) else {
            fatalError("Failed to load dental_patch_encoder.mlmodelc")
        }
        self.model = m

        if let embURL = bundle.url(forResource: "text_embeddings", withExtension: "bin") {
            (self.textEmb, self.labels) = Self.loadTextEmbeddings(url: embURL)
        } else {
            self.textEmb = nil
            self.labels  = nil
        }
    }

    // MARK: - Public inference

    /// Predict per-patch anatomical labels for a point cloud.
    ///
    /// - Parameter xyz: Flat Float array, N*3 row-major [x0,y0,z0, x1,y1,z1, ...]
    public func predict(xyz inputXYZ: [Float]) throws -> [PatchPrediction] {
        var xyz = inputXYZ
        PointOps.normalise(xyz: &xyz)

        let grouped = PointOps.group(xyz: xyz, numGroup: numGroup, groupSize: groupSize)
        let G = grouped.G
        let M = grouped.M

        let patchEmb = try runModel(neighborhood: grouped.neighborhood,
                                    centers: grouped.centers, G: G, M: M)

        guard let textEmb, let labels else { throw InferenceError.textEmbeddingsNotLoaded }

        let K      = labels.count
        var scores = [Float](repeating: 0, count: G * K)

        // scores = patchEmb (G x D)  @  textEmb.T (D x K)  →  (G x K)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(G), Int32(K), Int32(clipDim),
                    1.0,
                    patchEmb, Int32(clipDim),
                    textEmb,  Int32(clipDim),
                    0.0,
                    &scores,  Int32(K))

        return (0..<G).map { g in
            let rowStart = g * K
            var bestIdx  = rowStart
            for k in (rowStart + 1)..<(rowStart + K) {
                if scores[k] > scores[bestIdx] { bestIdx = k }
            }
            let labelIdx = bestIdx - rowStart
            return PatchPrediction(
                patchIndex: g,
                label:      labels[labelIdx],
                score:      scores[bestIdx],
                centerXYZ: (grouped.centers[g * 3],
                             grouped.centers[g * 3 + 1],
                             grouped.centers[g * 3 + 2])
            )
        }
    }

    /// Encode a point cloud into per-patch embeddings without classification.
    ///
    /// Returns both normalised and original-scale centres so callers can map
    /// patch results back onto the original model coordinate space.
    ///
    /// - Parameter xyz: Flat Float array, N*3 row-major [x0,y0,z0, x1,y1,z1, ...]
    public func encodePatches(xyz inputXYZ: [Float]) throws -> PatchEmbeddingOutput {
        let N = inputXYZ.count / 3

        // Replicate PointOps.normalise while capturing mean + scale for inversion
        var xyz = inputXYZ
        var mx: Float = 0, my: Float = 0, mz: Float = 0
        for i in 0..<N { mx += xyz[i*3]; my += xyz[i*3+1]; mz += xyz[i*3+2] }
        let fn = Float(N)
        mx /= fn; my /= fn; mz /= fn
        for i in 0..<N { xyz[i*3] -= mx; xyz[i*3+1] -= my; xyz[i*3+2] -= mz }
        var maxAbs: Float = 0
        for v in xyz { maxAbs = max(maxAbs, abs(v)) }
        var scale = 1.0 / (maxAbs + 1e-8)
        vDSP_vsmul(xyz, 1, &scale, &xyz, 1, vDSP_Length(xyz.count))

        let grouped = PointOps.group(xyz: xyz, numGroup: numGroup, groupSize: groupSize)
        let G = grouped.G
        let M = grouped.M

        let embeddings = try runModel(neighborhood: grouped.neighborhood,
                                      centers: grouped.centers, G: G, M: M)

        let invScale = 1.0 / scale
        var denorm = [Float](repeating: 0, count: G * 3)
        for g in 0..<G {
            denorm[g*3+0] = grouped.centers[g*3+0] * invScale + mx
            denorm[g*3+1] = grouped.centers[g*3+1] * invScale + my
            denorm[g*3+2] = grouped.centers[g*3+2] * invScale + mz
        }

        return PatchEmbeddingOutput(
            groupCount: G,
            centers: grouped.centers,
            denormalizedCenters: denorm,
            embeddings: embeddings,
            embeddingDim: clipDim
        )
    }

    // MARK: - Private

    private func runModel(
        neighborhood: [Float], centers: [Float], G: Int, M: Int
    ) throws -> [Float] {
        let neighArr = try MLMultiArray(shape: [G, M, 3] as [NSNumber], dataType: .float32)
        let centArr  = try MLMultiArray(shape: [G, 3]   as [NSNumber], dataType: .float32)

        neighborhood.withUnsafeBufferPointer {
            neighArr.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: $0.count * 4)
        }
        centers.withUnsafeBufferPointer {
            centArr.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: $0.count * 4)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "neighborhood": MLFeatureValue(multiArray: neighArr),
            "centers":      MLFeatureValue(multiArray: centArr),
        ])
        let output = try model.prediction(from: input)

        guard let outArr = output.featureValue(for: "patch_embeddings")?.multiArrayValue
        else { throw InferenceError.missingOutput }

        let count = G * clipDim
        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBufferPointer {
            $0.baseAddress!.initialize(
                from: outArr.dataPointer.assumingMemoryBound(to: Float.self),
                count: count)
        }
        return result
    }

    /// Binary format (written by save_text_emb_bin.py):
    ///   Int32 LE : K (number of labels)
    ///   Int32 LE : D (embedding dim, e.g. 1280)
    ///   K*D*4 bytes : float32 LE embeddings, row-major
    ///   Remaining   : label strings, UTF-8, newline-separated
    private static func loadTextEmbeddings(url: URL) -> ([Float], [String]) {
        guard let data = try? Data(contentsOf: url) else {
            fatalError("Cannot read text_embeddings.bin")
        }
        var offset = 0

        func readInt32() -> Int {
            let v = data.withUnsafeBytes {
                $0.load(fromByteOffset: offset, as: Int32.self)
            }
            offset += 4
            return Int(v.littleEndian)
        }

        let K = readInt32()
        let D = readInt32()
        let floatCount = K * D

        var emb = [Float](repeating: 0, count: floatCount)
        emb.withUnsafeMutableBytes {
            data.copyBytes(to: $0, from: offset..<(offset + floatCount * 4))
        }
        offset += floatCount * 4

        let labelStr = String(data: data.subdata(in: offset..<data.count),
                              encoding: .utf8) ?? ""
        let labels = labelStr.split(separator: "\n", omittingEmptySubsequences: true)
                              .map(String.init)

        precondition(labels.count == K, "Expected \(K) labels, got \(labels.count)")
        return (emb, labels)
    }

    public enum InferenceError: Error {
        case missingOutput
        case textEmbeddingsNotLoaded
    }
}
