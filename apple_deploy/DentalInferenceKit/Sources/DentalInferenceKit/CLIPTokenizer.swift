/// CLIP BPE tokenizer — pure Swift, no Python dependency.
///
/// Replicates the byte-level BPE used by open_clip / CLIP ViT-bigG-14:
///   1. Encode input string as UTF-8 bytes
///   2. Map each byte to a unicode character via the byte→unicode table
///   3. Split on whitespace; mark word end with </w> suffix on last char-group
///   4. Apply BPE merge rules (clip_merges.txt) iteratively
///   5. Map tokens to IDs via clip_vocab.json
///   6. Prepend SOT (49406), append EOT (49407), pad to 77
///
/// Required bundle resources (DentalInferenceKit/Resources/):
///   clip_vocab.json   — {"token_string": int_id, ...}
///   clip_merges.txt   — one merge rule per line "tok_a tok_b"

import Foundation

public final class CLIPTokenizer: @unchecked Sendable {

    // MARK: - Constants

    public static let contextLength = 77
    public static let sotToken: Int32 = 49406
    public static let eotToken: Int32 = 49407

    // MARK: - Private state

    private let vocab:      [String: Int32]
    private let mergeRanks: [Pair: Int]
    private let byteToUni:  [UInt8: Character]
    private var bpeCache:   [String: [String]] = [:]

    // MARK: - Init

    public init() {
        guard let vocabURL = ResourceLocator.url(forResource: "clip_vocab", withExtension: "json"),
              let vocabData = try? Data(contentsOf: vocabURL),
              let rawVocab  = try? JSONSerialization.jsonObject(with: vocabData) as? [String: Int]
        else { fatalError("clip_vocab.json not found in bundle") }
        self.vocab = rawVocab.mapValues { Int32($0) }

        guard let mergesURL = ResourceLocator.url(forResource: "clip_merges", withExtension: "txt"),
              let mergesText = try? String(contentsOf: mergesURL, encoding: .utf8)
        else { fatalError("clip_merges.txt not found in bundle") }

        var ranks: [Pair: Int] = [:]
        for (idx, line) in mergesText
                .split(separator: "\n", omittingEmptySubsequences: true)
                .enumerated() {
            let parts = line.split(separator: " ", maxSplits: 1)
            if parts.count == 2 {
                ranks[Pair(String(parts[0]), String(parts[1]))] = idx
            }
        }
        self.mergeRanks = ranks
        self.byteToUni  = Self.buildByteToUnicode()
    }

    // MARK: - Public API

    /// Tokenise `text` → Int32 array of length 77.
    public func tokenize(_ text: String) -> [Int32] {
        let cleaned = text.trimmingCharacters(in: .whitespaces).lowercased()
        let tokens  = bpeTokens(for: cleaned)

        var ids = [Int32](repeating: 0, count: Self.contextLength)
        ids[0]  = Self.sotToken

        let limit = min(tokens.count, Self.contextLength - 2)
        for i in 0..<limit {
            ids[i + 1] = vocab[tokens[i]] ?? 0
        }
        ids[limit + 1] = Self.eotToken
        return ids
    }

    // MARK: - BPE pipeline

    private func bpeTokens(for text: String) -> [String] {
        let words = text.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        var result: [String] = []
        for word in words {
            let bytes = Array(word.utf8)
            var chars = bytes.map { byteToUni[$0].map(String.init) ?? "?" }
            if !chars.isEmpty { chars[chars.count - 1] += "</w>" }
            result.append(contentsOf: bpe(chars))
        }
        return result
    }

    private func bpe(_ chars: [String]) -> [String] {
        let key = chars.joined(separator: "\u{2581}")   // unique separator
        if let cached = bpeCache[key] { return cached }

        var word = chars
        while word.count > 1 {
            var bestRank = Int.max
            var bestPair: Pair? = nil
            for i in 0..<(word.count - 1) {
                let p = Pair(word[i], word[i + 1])
                if let r = mergeRanks[p], r < bestRank {
                    bestRank = r
                    bestPair = p
                }
            }
            guard let pair = bestPair else { break }

            let combined = pair.first + pair.second
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1,
                   word[i] == pair.first,
                   word[i + 1] == pair.second {
                    merged.append(combined)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
        }

        bpeCache[key] = word
        return word
    }

    // MARK: - Byte-to-unicode table (mirrors Python's CLIP bytes_to_unicode)

    private static func buildByteToUnicode() -> [UInt8: Character] {
        var bs: [Int] = []
        bs += (Int(UInt8(ascii: "!"))...Int(UInt8(ascii: "~"))).map { $0 }
        bs += (0xA1...0xAC).map { $0 }
        bs += (0xAE...0xFF).map { $0 }

        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }

        var table: [UInt8: Character] = [:]
        for (b, c) in zip(bs, cs) {
            if let scalar = Unicode.Scalar(c) {
                table[UInt8(b)] = Character(scalar)
            }
        }
        return table
    }

    // MARK: - Pair

    private struct Pair: Hashable {
        let first: String
        let second: String
        init(_ a: String, _ b: String) { first = a; second = b }
    }
}
