/// Text-driven dental patch search — the main public API for query-by-text.
///
/// Two usage modes:
///
///   A) Search over pre-computed embeddings (low latency, cache patch emb):
///      let hits = try PatchSearchEngine.shared.search(
///          query:           "mandibular condyle",
///          patchEmbeddings: patchEmb,   // [Float] G*1280, L2-normalised
///          centers:         centers,    // [Float] G*3
///          topK: 5)
///
///   B) End-to-end from raw point cloud:
///      let hits = try PatchSearchEngine.shared.predict(
///          xyz:   flatXYZ,              // [Float] N*3
///          query: "mandibular condyle",
///          topK:  5)

import Accelerate
import CoreML
import Foundation

// MARK: - Result type

public struct SearchResult: Sendable {
    public let rank:      Int
    public let patch:     Int
    public let score:     Float
    public let centerXYZ: (x: Float, y: Float, z: Float)
}

// MARK: - Engine

public final class PatchSearchEngine: @unchecked Sendable {

    public static let shared = PatchSearchEngine()

    private let encoder = CLIPTextEncoder.shared
    private let dim     = 1280

    private init() {}

    // MARK: - Mode A: search over pre-computed patch embeddings

    /// Rank G patches by cosine similarity to `query`.
    ///
    /// - Parameters:
    ///   - query:           Free-text anatomical description.
    ///   - patchEmbeddings: Flat Float array, G×1280 row-major, L2-normalised.
    ///   - centers:         Flat Float array, G×3 row-major patch centres.
    ///   - topK:            Number of top results to return.
    public func search(
        query:           String,
        patchEmbeddings: [Float],
        centers:         [Float],
        topK:            Int = 5
    ) throws -> [SearchResult] {
        let G = patchEmbeddings.count / dim
        precondition(patchEmbeddings.count == G * dim, "patchEmbeddings must be G*\(dim)")
        precondition(centers.count == G * 3,           "centers must be G*3")

        let textEmb = try encoder.encode(query)   // (1280,) L2-normalised

        var scores = [Float](repeating: 0, count: G)
        textEmb.withUnsafeBufferPointer { tBuf in
            patchEmbeddings.withUnsafeBufferPointer { pBuf in
                cblas_sgemv(
                    CblasRowMajor, CblasNoTrans,
                    Int32(G), Int32(dim),
                    1.0,
                    pBuf.baseAddress!, Int32(dim),
                    tBuf.baseAddress!, 1,
                    0.0,
                    &scores, 1)
            }
        }

        return topKIndices(scores, k: topK).enumerated().map { rank, i in
            SearchResult(
                rank:      rank + 1,
                patch:     i,
                score:     scores[i],
                centerXYZ: (centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2]))
        }
    }

    // MARK: - Mode B: end-to-end pipeline

    /// Run point encoding + text search in one call.
    ///
    /// - Parameters:
    ///   - xyz:   Flat Float array, N×3 row-major [x0,y0,z0, ...].
    ///   - query: Free-text anatomical description.
    ///   - topK:  Number of results to return.
    public func predict(
        xyz:   [Float],
        query: String,
        topK:  Int = 5
    ) throws -> [SearchResult] {
        let inference = DentalInference.shared

        var mutableXYZ = xyz
        PointOps.normalise(xyz: &mutableXYZ)

        let grouped = PointOps.group(
            xyz:       mutableXYZ,
            numGroup:  inference.numGroup,
            groupSize: inference.groupSize)

        let patchEmb = try runPatchEncoder(
            neighborhood: grouped.neighborhood,
            centers:      grouped.centers,
            G:            grouped.G,
            M:            grouped.M)

        return try search(
            query:           query,
            patchEmbeddings: patchEmb,
            centers:         grouped.centers,
            topK:            topK)
    }

    // MARK: - Private helpers

    private func runPatchEncoder(
        neighborhood: [Float],
        centers:      [Float],
        G: Int,
        M: Int
    ) throws -> [Float] {
        let modelURL = try ResourceLocator.coreMLModelURL(named: "dental_patch_encoder")

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        let model = try MLModel(contentsOf: modelURL, configuration: cfg)

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
        else { throw SearchError.missingOutput }

        let count = G * dim
        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBufferPointer {
            $0.baseAddress!.initialize(
                from: outArr.dataPointer.assumingMemoryBound(to: Float.self),
                count: count)
        }
        return result
    }

    private func topKIndices(_ scores: [Float], k: Int) -> [Int] {
        scores.indices
            .sorted { scores[$0] > scores[$1] }
            .prefix(min(k, scores.count))
            .map { $0 }
    }

    public enum SearchError: Error {
        case missingOutput
    }
}
