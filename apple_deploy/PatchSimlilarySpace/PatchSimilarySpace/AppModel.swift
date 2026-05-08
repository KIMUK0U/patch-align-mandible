//
//  AppModel.swift
//  PatchSimlilarySpace
//
//  Created by 木村亘汰 on 2026/04/24.
//

import DentalInferenceKit
import Foundation
import Observation
import RealityKit
import SwiftUI

nonisolated struct PatchRecord: Identifiable, Sendable {
    let id: Int
    let normalizedCenter: SIMD3<Float>
    let modelLocalCenter: SIMD3<Float>
    let rootLocalCenter: SIMD3<Float>
}

nonisolated struct PatchDatabank: Sendable {
    let embeddings: [Float]
    let normalizedCenters: [Float]
    let records: [PatchRecord]
    let embeddingDim: Int
}

nonisolated struct SamplingOutput: Sendable {
    let sampledXYZ: [Float]
    let sampledCount: Int
}

nonisolated struct ModelSource: Equatable {
    let url: URL
    let displayName: String
    let isSecurityScoped: Bool
}

enum PipelineStepID: Int, CaseIterable, Identifiable {
    case model = 1
    case patches = 2
    case text = 3

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .model: "USDZ"
        case .patches: "Patches"
        case .text: "Text"
        }
    }
}

enum PipelineStepStatus: Equatable {
    case waiting
    case running
    case done
    case failed
}

struct PipelineStepState: Identifiable, Equatable {
    let id: PipelineStepID
    var status: PipelineStepStatus
}

enum AppPipelineError: LocalizedError {
    case missingDefaultUSDZ
    case missingRealityRoot
    case noMeshVertices
    case emptyQuery
    case missingDatabank
    case missingPatch(Int)

    var errorDescription: String? {
        switch self {
        case .missingDefaultUSDZ:
            return "Mandible.usdz was not found in the app bundle."
        case .missingRealityRoot:
            return "Open the immersive space before loading a model."
        case .noMeshVertices:
            return "No mesh vertices were found in the USDZ file."
        case .emptyQuery:
            return "Enter text before searching."
        case .missingDatabank:
            return "Patch databank is not ready."
        case let .missingPatch(patch):
            return "Patch \(patch) was not found in the databank."
        }
    }
}

/// Maintains app-wide state.
@MainActor
@Observable
final class AppModel {
    let immersiveSpaceID = "ImmersiveSpace"

    enum ImmersiveSpaceState {
        case closed
        case inTransition
        case open
    }

    var immersiveSpaceState = ImmersiveSpaceState.closed
    var steps: [PipelineStepState] = PipelineStepID.allCases.map {
        PipelineStepState(id: $0, status: .waiting)
    }
    var queryText = ""
    var isSearching = false
    var selectedModelName = "Mandible.usdz"
    var statusMessage: String?
    var errorMessage: String?
    var lastMatchText: String?

    @ObservationIgnored private var modelRoot: Entity?
    @ObservationIgnored private var loadedModelEntity: Entity?
    @ObservationIgnored private var activeMarkerEntity: Entity?
    @ObservationIgnored private var selectedSource: ModelSource?
    @ObservationIgnored private var databank: PatchDatabank?
    @ObservationIgnored private var pipelineTask: Task<Void, Never>?
    @ObservationIgnored private var pipelineRunID = 0

    var isReadyForQueries: Bool {
        status(for: .text) == .done && databank != nil
    }

    func attachRealityRoot(_ root: Entity) {
        modelRoot = root
        if selectedSource == nil {
            selectedSource = defaultModelSource()
        }
        startPipeline()
    }

    func detachRealityRoot() {
        pipelineTask?.cancel()
        pipelineTask = nil
        modelRoot = nil
        loadedModelEntity = nil
        activeMarkerEntity = nil
    }

    func selectUSDZ(url: URL) {
        selectedSource = ModelSource(
            url: url,
            displayName: url.lastPathComponent,
            isSecurityScoped: true
        )
        selectedModelName = url.lastPathComponent
        startPipeline()
    }

    func reloadCurrentModel() {
        startPipeline()
    }

    func searchCurrentQuery() async {
        let query = queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else {
            errorMessage = AppPipelineError.emptyQuery.localizedDescription
            return
        }
        guard let databank else {
            errorMessage = AppPipelineError.missingDatabank.localizedDescription
            return
        }

        isSearching = true
        errorMessage = nil
        lastMatchText = nil
        defer { isSearching = false }

        do {
            let hit = try await Task.detached(priority: .userInitiated) {
                try PatchSearchEngine.shared.search(
                    query: query,
                    patchEmbeddings: databank.embeddings,
                    centers: databank.normalizedCenters,
                    topK: 1
                ).first
            }.value

            guard let hit else { throw AppPipelineError.missingDatabank }
            guard let record = databank.records.first(where: { $0.id == hit.patch }) else {
                throw AppPipelineError.missingPatch(hit.patch)
            }

            placeMarker(at: record.rootLocalCenter)
            lastMatchText = "Patch \(record.id)  score \(String(format: "%.3f", hit.score))"
        } catch {
            clearMarker()
            errorMessage = error.localizedDescription
        }
    }

    func status(for id: PipelineStepID) -> PipelineStepStatus {
        steps.first(where: { $0.id == id })?.status ?? .waiting
    }

    private func defaultModelSource() -> ModelSource? {
        guard let url = Bundle.main.url(forResource: "Mandible", withExtension: "usdz") else {
            return nil
        }
        return ModelSource(url: url, displayName: "Mandible.usdz", isSecurityScoped: false)
    }

    private func startPipeline() {
        pipelineTask?.cancel()
        pipelineRunID += 1
        let runID = pipelineRunID

        pipelineTask = Task { [weak self] in
            await self?.runPipeline(runID: runID)
        }
    }

    private func runPipeline(runID: Int) async {
        guard let source = selectedSource ?? defaultModelSource() else {
            fail(step: .model, message: AppPipelineError.missingDefaultUSDZ.localizedDescription)
            return
        }
        guard let modelRoot else {
            fail(step: .model, message: AppPipelineError.missingRealityRoot.localizedDescription)
            return
        }

        resetPipelineState()
        selectedModelName = source.displayName
        setStep(.model, .running)

        do {
            let didStartAccessing = source.isSecurityScoped && source.url.startAccessingSecurityScopedResource()
            defer {
                if didStartAccessing {
                    source.url.stopAccessingSecurityScopedResource()
                }
            }

            let entity = try await Entity(contentsOf: source.url)
            try Task.checkCancellation()
            guard runID == pipelineRunID else { return }

            let rawPoints = try Self.extractModelPoints(from: entity)
            let centroid = Self.boundsCenter(for: rawPoints)
            configureLoadedModel(entity, centroid: centroid, in: modelRoot)
            setStep(.model, .done)

            setStep(.patches, .running)
            let sampling = try await Task.detached(priority: .userInitiated) {
                try Self.samplePoints(rawPoints)
            }.value
            try Task.checkCancellation()

            let encoded = try await Task.detached(priority: .userInitiated) {
                try DentalInference.shared.encodePatches(xyz: sampling.sampledXYZ)
            }.value
            try Task.checkCancellation()
            guard runID == pipelineRunID else { return }

            databank = Self.makeDatabank(from: encoded, modelCentroid: centroid)
            statusMessage = "\(sampling.sampledCount) points / \(encoded.groupCount) patches"
            setStep(.patches, .done)

            setStep(.text, .running)
            try await Task.detached(priority: .userInitiated) {
                try CLIPTextEncoder.shared.warmUp()
            }.value
            try Task.checkCancellation()
            guard runID == pipelineRunID else { return }

            setStep(.text, .done)
            statusMessage = "Ready"
        } catch is CancellationError {
            return
        } catch {
            let failedStep: PipelineStepID = status(for: .patches) == .running ? .patches : status(for: .text) == .running ? .text : .model
            fail(step: failedStep, message: error.localizedDescription)
        }
    }

    private func resetPipelineState() {
        for index in steps.indices {
            steps[index].status = .waiting
        }
        databank = nil
        statusMessage = nil
        errorMessage = nil
        lastMatchText = nil
        clearMarker()
        loadedModelEntity?.removeFromParent()
        loadedModelEntity = nil
    }

    private func setStep(_ id: PipelineStepID, _ status: PipelineStepStatus) {
        guard let index = steps.firstIndex(where: { $0.id == id }) else { return }
        steps[index].status = status
    }

    private func fail(step: PipelineStepID, message: String) {
        setStep(step, .failed)
        statusMessage = nil
        errorMessage = message
    }

    private func configureLoadedModel(_ entity: Entity, centroid: SIMD3<Float>, in modelRoot: Entity) {
        modelRoot.children.removeAll()
        modelRoot.position = SIMD3<Float>(0, 1.5, -1)
        modelRoot.orientation = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))
        entity.position = -centroid
        modelRoot.addChild(entity)
        loadedModelEntity = entity
    }

    private func placeMarker(at rootLocalCenter: SIMD3<Float>) {
        clearMarker()
        let mesh = MeshResource.generateSphere(radius: 0.01)
        let material = SimpleMaterial(color: .red, roughness: 0.35, isMetallic: false)
        let marker = ModelEntity(mesh: mesh, materials: [material])
        marker.name = "PatchSearchMarker"
        marker.position = rootLocalCenter
        modelRoot?.addChild(marker)
        activeMarkerEntity = marker
    }

    private func clearMarker() {
        activeMarkerEntity?.removeFromParent()
        activeMarkerEntity = nil
    }

    private static func extractModelPoints(from root: Entity) throws -> [SIMD3<Float>] {
        var points: [SIMD3<Float>] = []

        func visit(_ entity: Entity) {
            if let modelEntity = entity as? ModelEntity,
               let mesh = modelEntity.model?.mesh {
                for model in mesh.contents.models {
                    for part in model.parts {
                        for position in part.positions {
                            points.append(modelEntity.convert(position: position, to: root))
                        }
                    }
                }
            }

            for child in entity.children {
                visit(child)
            }
        }

        visit(root)
        guard !points.isEmpty else { throw AppPipelineError.noMeshVertices }
        return points
    }

    private nonisolated static func samplePoints(_ points: [SIMD3<Float>]) throws -> SamplingOutput {
        var xyz: [Float] = []
        xyz.reserveCapacity(points.count * 3)
        for point in points {
            xyz.append(point.x)
            xyz.append(point.y)
            xyz.append(point.z)
        }

        let sampled = try PointOps.voxelFPSDownsample(xyz: xyz, targetCount: 2048)
        return SamplingOutput(sampledXYZ: sampled, sampledCount: sampled.count / 3)
    }

    private nonisolated static func makeDatabank(
        from encoded: PatchEmbeddingOutput,
        modelCentroid: SIMD3<Float>
    ) -> PatchDatabank {
        var records: [PatchRecord] = []
        records.reserveCapacity(encoded.groupCount)

        for patchID in 0..<encoded.groupCount {
            let normalizedCenter = SIMD3<Float>(
                encoded.centers[patchID * 3],
                encoded.centers[patchID * 3 + 1],
                encoded.centers[patchID * 3 + 2]
            )
            let modelLocalCenter = SIMD3<Float>(
                encoded.denormalizedCenters[patchID * 3],
                encoded.denormalizedCenters[patchID * 3 + 1],
                encoded.denormalizedCenters[patchID * 3 + 2]
            )
            records.append(
                PatchRecord(
                    id: patchID,
                    normalizedCenter: normalizedCenter,
                    modelLocalCenter: modelLocalCenter,
                    rootLocalCenter: modelLocalCenter - modelCentroid
                )
            )
        }

        return PatchDatabank(
            embeddings: encoded.embeddings,
            normalizedCenters: encoded.centers,
            records: records,
            embeddingDim: encoded.embeddingDim
        )
    }

    private static func boundsCenter(for points: [SIMD3<Float>]) -> SIMD3<Float> {
        var minPoint = SIMD3<Float>(repeating: Float.infinity)
        var maxPoint = SIMD3<Float>(repeating: -Float.infinity)
        for point in points {
            minPoint = min(minPoint, point)
            maxPoint = max(maxPoint, point)
        }
        return (minPoint + maxPoint) * 0.5
    }
}
