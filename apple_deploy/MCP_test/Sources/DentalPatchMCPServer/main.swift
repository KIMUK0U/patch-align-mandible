import DentalInferenceKit
import Foundation
import MCP
import ModelIO
import simd

private let serverVersion = "0.1.0"

@main
struct DentalPatchMCPServer {
    static func main() async throws {
        let service = DentalPatchService()
        let server = Server(
            name: "dental-patch-mcp",
            version: serverVersion,
            instructions: """
            Dental patch search tools for research use. Load a local USDZ mandible model first,
            then search patch ids by anatomical text or describe an existing patch id by text-bank similarity.
            Results are similarity scores, not medical diagnoses.
            """,
            capabilities: .init(
                tools: .init(listChanged: false)
            )
        )

        await registerToolHandlers(server: server, service: service)

        let transport = StdioTransport()
        try await server.start(transport: transport)
        await server.waitUntilCompleted()
        await server.stop()
    }
}

private func registerToolHandlers(server: Server, service: DentalPatchService) async {
    await server.withMethodHandler(ListTools.self) { _ in
        ListTools.Result(tools: [
            Tool(
                name: "load_dental_model",
                title: "Load Dental USDZ Model",
                description: "Load a local USDZ dental model, sample its mesh vertices, encode patch embeddings, and cache them under a model id.",
                inputSchema: objectSchema(
                    properties: [
                        "usdz_path": stringProperty("Absolute or ~/ relative path to a local .usdz file."),
                        "model_id": stringProperty("Optional caller-chosen id. If omitted, the server creates one from the file name."),
                    ],
                    required: ["usdz_path"]
                ),
                annotations: .init(
                    title: "Load dental model",
                    readOnlyHint: false,
                    destructiveHint: false,
                    idempotentHint: false,
                    openWorldHint: false
                )
            ),
            Tool(
                name: "search_patch_by_text",
                title: "Search Patch By Text",
                description: "Search cached patch embeddings by anatomical text and return ranked patch ids with similarity scores and centers.",
                inputSchema: objectSchema(
                    properties: [
                        "model_id": stringProperty("Model id returned by load_dental_model."),
                        "query": stringProperty("Anatomical text query, for example left mandibular condyle."),
                        "top_k": integerProperty("Number of ranked patches to return. Default: 5."),
                    ],
                    required: ["model_id", "query"]
                ),
                annotations: .init(
                    title: "Search patch by text",
                    readOnlyHint: true,
                    destructiveHint: false,
                    idempotentHint: true,
                    openWorldHint: false
                )
            ),
            Tool(
                name: "describe_patch",
                title: "Describe Patch",
                description: "Compare one cached patch embedding against the registered text bank and return likely text labels.",
                inputSchema: objectSchema(
                    properties: [
                        "model_id": stringProperty("Model id returned by load_dental_model."),
                        "patch_id": integerProperty("Patch id to describe."),
                        "top_k": integerProperty("Number of text-bank candidates to return. Default: 5."),
                    ],
                    required: ["model_id", "patch_id"]
                ),
                annotations: .init(
                    title: "Describe patch",
                    readOnlyHint: true,
                    destructiveHint: false,
                    idempotentHint: true,
                    openWorldHint: false
                )
            ),
            Tool(
                name: "list_loaded_models",
                title: "List Loaded Models",
                description: "List USDZ models currently cached in this MCP server process.",
                inputSchema: objectSchema(properties: [:]),
                annotations: .init(
                    title: "List loaded models",
                    readOnlyHint: true,
                    destructiveHint: false,
                    idempotentHint: true,
                    openWorldHint: false
                )
            ),
        ])
    }

    await server.withMethodHandler(CallTool.self) { params in
        do {
            let resultText: String
            switch params.name {
            case "load_dental_model":
                let usdzPath = try requiredString(params.arguments, "usdz_path")
                let modelID = optionalString(params.arguments, "model_id")
                resultText = try await jsonText(service.loadModel(usdzPath: usdzPath, requestedModelID: modelID))

            case "search_patch_by_text":
                let modelID = try requiredString(params.arguments, "model_id")
                let query = try requiredString(params.arguments, "query")
                let topK = try optionalInt(params.arguments, "top_k", default: 5)
                resultText = try await jsonText(service.searchPatchByText(modelID: modelID, query: query, topK: topK))

            case "describe_patch":
                let modelID = try requiredString(params.arguments, "model_id")
                let patchID = try requiredInt(params.arguments, "patch_id")
                let topK = try optionalInt(params.arguments, "top_k", default: 5)
                resultText = try await jsonText(service.describePatch(modelID: modelID, patchID: patchID, topK: topK))

            case "list_loaded_models":
                resultText = try await jsonText(service.listLoadedModels())

            default:
                throw ToolArgumentError.unknownTool(params.name)
            }

            return CallTool.Result(
                content: [.text(text: resultText, annotations: nil, _meta: nil)],
                isError: false
            )
        } catch {
            let payload = ErrorOutput(error: error.localizedDescription)
            let resultText = (try? jsonText(payload)) ?? #"{"error":"Tool failed"}"#
            return CallTool.Result(
                content: [.text(text: resultText, annotations: nil, _meta: nil)],
                isError: true
            )
        }
    }
}

// MARK: - Service

private actor DentalPatchService {
    private var models: [String: CachedModel] = [:]

    func loadModel(usdzPath: String, requestedModelID: String?) async throws -> LoadModelOutput {
        log("load_dental_model: validating USDZ path")
        let url = try validateUSDZPath(usdzPath)
        log("load_dental_model: loading USDZ mesh")
        let rawPoints = try USDZPointLoader.loadPoints(from: url)
        log("load_dental_model: sampling \(rawPoints.count) mesh vertices")
        let sampledXYZ = try samplePoints(rawPoints)
        log("load_dental_model: encoding \(sampledXYZ.count / 3) sampled points")
        let encoded = try DentalInference.shared.encodePatches(xyz: sampledXYZ)
        log("load_dental_model: caching \(encoded.groupCount) patches")
        let databank = PatchDatabank(encoded: encoded)
        let modelID = makeModelID(requested: requestedModelID, url: url)

        let cached = CachedModel(
            id: modelID,
            sourcePath: url.path,
            sampledPointCount: sampledXYZ.count / 3,
            patchCount: encoded.groupCount,
            embeddingDim: encoded.embeddingDim,
            loadedAt: ISO8601DateFormatter().string(from: Date()),
            databank: databank
        )
        models[modelID] = cached

        return LoadModelOutput(
            model_id: modelID,
            source_path: url.path,
            sampled_point_count: cached.sampledPointCount,
            patch_count: cached.patchCount,
            embedding_dim: cached.embeddingDim,
            text_bank_count: DentalInference.shared.textBankCount,
            loaded_at: cached.loadedAt
        )
    }

    func searchPatchByText(modelID: String, query: String, topK: Int) async throws -> SearchPatchOutput {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ToolArgumentError.emptyArgument("query")
        }
        let model = try cachedModel(id: modelID)
        let limit = clampTopK(topK, max: model.patchCount)

        let hits = try PatchSearchEngine.shared.search(
            query: query,
            patchEmbeddings: model.databank.embeddings,
            centers: model.databank.normalizedCenters,
            topK: limit
        )

        let results = hits.map { hit in
            PatchHitOutput(
                rank: hit.rank,
                patch_id: hit.patch,
                score: hit.score,
                normalized_center_xyz: vector3(model.databank.normalizedCenters, hit.patch),
                model_center_xyz: vector3(model.databank.denormalizedCenters, hit.patch)
            )
        }

        return SearchPatchOutput(
            model_id: model.id,
            query: query,
            top_k: limit,
            results: results,
            note: "Similarity scores are research outputs, not diagnostic findings."
        )
    }

    func describePatch(modelID: String, patchID: Int, topK: Int) async throws -> DescribePatchOutput {
        let model = try cachedModel(id: modelID)
        guard (0..<model.patchCount).contains(patchID) else {
            throw ToolArgumentError.patchOutOfRange(patchID: patchID, patchCount: model.patchCount)
        }

        let limit = clampTopK(topK, max: max(DentalInference.shared.textBankCount, 1))
        let start = patchID * model.embeddingDim
        let embedding = Array(model.databank.embeddings[start..<(start + model.embeddingDim)])
        let matches = try DentalInference.shared.describePatchEmbedding(embedding, topK: limit)

        return DescribePatchOutput(
            model_id: model.id,
            patch_id: patchID,
            normalized_center_xyz: vector3(model.databank.normalizedCenters, patchID),
            model_center_xyz: vector3(model.databank.denormalizedCenters, patchID),
            top_k: limit,
            text_candidates: matches.map {
                TextCandidateOutput(
                    rank: $0.rank,
                    text_index: $0.textIndex,
                    label: $0.label,
                    score: $0.score
                )
            },
            note: "Text candidates come from text_embeddings.bin and should be interpreted as similarity matches."
        )
    }

    func listLoadedModels() -> ListModelsOutput {
        let summaries = models.values
            .sorted { $0.id < $1.id }
            .map {
                ModelSummaryOutput(
                    model_id: $0.id,
                    source_path: $0.sourcePath,
                    sampled_point_count: $0.sampledPointCount,
                    patch_count: $0.patchCount,
                    embedding_dim: $0.embeddingDim,
                    loaded_at: $0.loadedAt
                )
            }

        return ListModelsOutput(models: summaries)
    }

    private func cachedModel(id: String) throws -> CachedModel {
        guard let model = models[id] else {
            throw ToolArgumentError.modelNotLoaded(id)
        }
        return model
    }
}

// MARK: - USDZ point loading

private enum USDZPointLoader {
    static func loadPoints(from url: URL) throws -> [SIMD3<Float>] {
        let asset = MDLAsset(url: url)
        var points: [SIMD3<Float>] = []

        for index in 0..<asset.count {
            visit(asset.object(at: index), parentTransform: matrix_identity_float4x4, points: &points)
        }

        guard !points.isEmpty else {
            throw ToolArgumentError.noMeshVertices
        }
        return points
    }

    private static func visit(
        _ object: MDLObject,
        parentTransform: simd_float4x4,
        points: inout [SIMD3<Float>]
    ) {
        let worldTransform = parentTransform * (object.transform?.matrix ?? matrix_identity_float4x4)

        if let mesh = object as? MDLMesh {
            appendPositions(from: mesh, transform: worldTransform, to: &points)
        }

        for child in object.children.objects {
            visit(child, parentTransform: worldTransform, points: &points)
        }
    }

    private static func appendPositions(
        from mesh: MDLMesh,
        transform: simd_float4x4,
        to points: inout [SIMD3<Float>]
    ) {
        guard
            let attribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributePosition),
            attribute.bufferIndex < mesh.vertexBuffers.count,
            let layout = mesh.vertexDescriptor.layouts[attribute.bufferIndex] as? MDLVertexBufferLayout,
            attribute.format == .float3 || attribute.format == .float4
        else {
            return
        }

        let mapped = mesh.vertexBuffers[attribute.bufferIndex].map()
        let base = mapped.bytes
        let stride = layout.stride
        let offset = attribute.offset

        for vertexIndex in 0..<mesh.vertexCount {
            let pointer = base
                .advanced(by: vertexIndex * stride + offset)
                .assumingMemoryBound(to: Float.self)
            let local = SIMD4<Float>(pointer[0], pointer[1], pointer[2], 1)
            let world = transform * local
            points.append(SIMD3<Float>(world.x, world.y, world.z))
        }
    }
}

// MARK: - Data model

private struct CachedModel: Sendable {
    let id: String
    let sourcePath: String
    let sampledPointCount: Int
    let patchCount: Int
    let embeddingDim: Int
    let loadedAt: String
    let databank: PatchDatabank
}

private struct PatchDatabank: Sendable {
    let embeddings: [Float]
    let normalizedCenters: [Float]
    let denormalizedCenters: [Float]
    let embeddingDim: Int
    let patchCount: Int

    init(encoded: PatchEmbeddingOutput) {
        self.embeddings = encoded.embeddings
        self.normalizedCenters = encoded.centers
        self.denormalizedCenters = encoded.denormalizedCenters
        self.embeddingDim = encoded.embeddingDim
        self.patchCount = encoded.groupCount
    }
}

private struct LoadModelOutput: Codable {
    let model_id: String
    let source_path: String
    let sampled_point_count: Int
    let patch_count: Int
    let embedding_dim: Int
    let text_bank_count: Int
    let loaded_at: String
}

private struct SearchPatchOutput: Codable {
    let model_id: String
    let query: String
    let top_k: Int
    let results: [PatchHitOutput]
    let note: String
}

private struct PatchHitOutput: Codable {
    let rank: Int
    let patch_id: Int
    let score: Float
    let normalized_center_xyz: [Float]
    let model_center_xyz: [Float]
}

private struct DescribePatchOutput: Codable {
    let model_id: String
    let patch_id: Int
    let normalized_center_xyz: [Float]
    let model_center_xyz: [Float]
    let top_k: Int
    let text_candidates: [TextCandidateOutput]
    let note: String
}

private struct TextCandidateOutput: Codable {
    let rank: Int
    let text_index: Int
    let label: String
    let score: Float
}

private struct ListModelsOutput: Codable {
    let models: [ModelSummaryOutput]
}

private struct ModelSummaryOutput: Codable {
    let model_id: String
    let source_path: String
    let sampled_point_count: Int
    let patch_count: Int
    let embedding_dim: Int
    let loaded_at: String
}

private struct ErrorOutput: Codable {
    let error: String
}

// MARK: - Helpers

private func samplePoints(_ points: [SIMD3<Float>]) throws -> [Float] {
    var xyz: [Float] = []
    xyz.reserveCapacity(points.count * 3)
    for point in points {
        xyz.append(point.x)
        xyz.append(point.y)
        xyz.append(point.z)
    }
    return try PointOps.voxelFPSDownsample(xyz: xyz, targetCount: 2048)
}

private func vector3(_ values: [Float], _ index: Int) -> [Float] {
    let start = index * 3
    return [values[start], values[start + 1], values[start + 2]]
}

private func clampTopK(_ topK: Int, max upperBound: Int) -> Int {
    Swift.max(1, Swift.min(topK, upperBound))
}

private func makeModelID(requested: String?, url: URL) -> String {
    if let requested, !requested.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        return requested
    }

    let base = url.deletingPathExtension().lastPathComponent
        .lowercased()
        .map { character -> Character in
            character.isLetter || character.isNumber ? character : "-"
        }
    let compactBase = String(base).split(separator: "-").joined(separator: "-")
    return "\(compactBase)-\(UUID().uuidString.prefix(8))"
}

private func validateUSDZPath(_ rawPath: String) throws -> URL {
    let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
        throw ToolArgumentError.emptyArgument("usdz_path")
    }

    let expanded: String
    if trimmed == "~" {
        expanded = FileManager.default.homeDirectoryForCurrentUser.path
    } else if trimmed.hasPrefix("~/") {
        expanded = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(String(trimmed.dropFirst(2)))
            .path
    } else {
        expanded = trimmed
    }

    let url = URL(fileURLWithPath: expanded)
    guard url.pathExtension.lowercased() == "usdz" else {
        throw ToolArgumentError.unsupportedFileType(url.pathExtension)
    }
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw ToolArgumentError.fileNotFound(url.path)
    }
    return url
}

private func jsonText<T: Encodable>(_ value: T) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    return String(decoding: try encoder.encode(value), as: UTF8.self)
}

private func objectSchema(properties: [String: Value], required: [String] = []) -> Value {
    var schema: [String: Value] = [
        "type": .string("object"),
        "properties": .object(properties),
        "additionalProperties": .bool(false),
    ]

    if !required.isEmpty {
        schema["required"] = .array(required.map { .string($0) })
    }

    return .object(schema)
}

private func stringProperty(_ description: String) -> Value {
    .object([
        "type": .string("string"),
        "description": .string(description),
    ])
}

private func integerProperty(_ description: String) -> Value {
    .object([
        "type": .string("integer"),
        "description": .string(description),
    ])
}

private func requiredString(_ arguments: [String: Value]?, _ name: String) throws -> String {
    guard let value = arguments?[name] else {
        throw ToolArgumentError.missingArgument(name)
    }
    guard let string = value.stringValue else {
        throw ToolArgumentError.invalidArgument(name, expected: "string")
    }
    let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
        throw ToolArgumentError.emptyArgument(name)
    }
    return trimmed
}

private func optionalString(_ arguments: [String: Value]?, _ name: String) -> String? {
    arguments?[name]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines)
}

private func requiredInt(_ arguments: [String: Value]?, _ name: String) throws -> Int {
    guard let value = arguments?[name] else {
        throw ToolArgumentError.missingArgument(name)
    }
    guard let int = intValue(value) else {
        throw ToolArgumentError.invalidArgument(name, expected: "integer")
    }
    return int
}

private func optionalInt(_ arguments: [String: Value]?, _ name: String, default defaultValue: Int) throws -> Int {
    guard let value = arguments?[name] else {
        return defaultValue
    }
    guard let int = intValue(value) else {
        throw ToolArgumentError.invalidArgument(name, expected: "integer")
    }
    return int
}

private func intValue(_ value: Value) -> Int? {
    if let int = value.intValue {
        return int
    }
    if let double = value.doubleValue {
        return Int(double)
    }
    if let string = value.stringValue {
        return Int(string)
    }
    return nil
}

private func log(_ message: String) {
    fputs("[DentalPatchMCP] \(message)\n", stderr)
}

private enum ToolArgumentError: LocalizedError {
    case missingArgument(String)
    case emptyArgument(String)
    case invalidArgument(String, expected: String)
    case unknownTool(String)
    case fileNotFound(String)
    case unsupportedFileType(String)
    case noMeshVertices
    case modelNotLoaded(String)
    case patchOutOfRange(patchID: Int, patchCount: Int)

    var errorDescription: String? {
        switch self {
        case .missingArgument(let name):
            return "Missing required argument: \(name)"
        case .emptyArgument(let name):
            return "Argument must not be empty: \(name)"
        case .invalidArgument(let name, let expected):
            return "Invalid argument \(name): expected \(expected)"
        case .unknownTool(let name):
            return "Unknown tool: \(name)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .unsupportedFileType(let ext):
            return "Unsupported file type: .\(ext). Only .usdz is supported."
        case .noMeshVertices:
            return "No mesh vertices were found in the USDZ file."
        case .modelNotLoaded(let id):
            return "Model is not loaded: \(id). Call load_dental_model first."
        case .patchOutOfRange(let patchID, let patchCount):
            return "Patch id \(patchID) is out of range. Valid range is 0...\(max(patchCount - 1, 0))."
        }
    }
}
