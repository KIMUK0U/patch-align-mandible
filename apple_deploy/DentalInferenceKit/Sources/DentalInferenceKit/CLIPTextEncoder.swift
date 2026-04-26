/// CoreML wrapper for the CLIP ViT-bigG-14 text encoder.
///
/// Combines CLIPTokenizer (BPE) with dental_clip_text.mlpackage to produce
/// a 1280-dim L2-normalised embedding for any input string.
///
/// Setup:
///   1. Run `clip_text/export_clip_coreml.py` → dental_clip_text.mlpackage
///   2. Place the package in DentalInferenceKit/Resources/
///
/// Usage:
///   let emb = try CLIPTextEncoder.shared.encode("condyle")  // [Float] length 1280

import CoreML
import Foundation

public final class CLIPTextEncoder: @unchecked Sendable {

    public static let shared = CLIPTextEncoder()

    public let embeddingDim = 1280

    private let model:     MLModel
    private let tokenizer: CLIPTokenizer

    // MARK: - Init

    private init() {
        let modelURL: URL
        do {
            modelURL = try ResourceLocator.coreMLModelURL(named: "dental_clip_text")
        } catch {
            fatalError(error.localizedDescription)
        }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        guard let m = try? MLModel(contentsOf: modelURL, configuration: cfg) else {
            fatalError("Failed to load dental_clip_text CoreML model")
        }
        self.model     = m
        self.tokenizer = CLIPTokenizer()
    }

    // MARK: - Public API

    /// Pre-warm the Neural Engine by encoding a dummy string.
    /// Call once before the first real encode to avoid first-use latency.
    public func warmUp() throws {
        _ = try encode("condyle")
    }

    /// Encode a free-text string → 1280-dim L2-normalised Float array.
    public func encode(_ text: String) throws -> [Float] {
        let ids = tokenizer.tokenize(text)   // [Int32], length 77

        let idsArr = try MLMultiArray(shape: [1, 77], dataType: .int32)
        ids.withUnsafeBufferPointer {
            idsArr.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: 77 * 4)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "token_ids": MLFeatureValue(multiArray: idsArr)
        ])
        let output = try model.prediction(from: input)

        guard let outArr = output.featureValue(for: "text_embedding")?.multiArrayValue
        else { throw EncoderError.missingOutput }

        var result = [Float](repeating: 0, count: embeddingDim)
        result.withUnsafeMutableBufferPointer {
            $0.baseAddress!.initialize(
                from: outArr.dataPointer.assumingMemoryBound(to: Float.self),
                count: embeddingDim)
        }
        return result
    }

    public enum EncoderError: Error {
        case missingOutput
    }
}
