//
//  ContentView.swift
//  PatchSimlilarySpace
//
//  Created by 木村亘汰 on 2026/04/24.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var appModel
    @State private var isImporterPresented = false

    var body: some View {
        @Bindable var appModel = appModel

        VStack(alignment: .leading, spacing: 20) {
            HStack(spacing: 12) {
                ToggleImmersiveSpaceButton()

                Button {
                    isImporterPresented = true
                } label: {
                    Label("USDZ", systemImage: "folder")
                }

                Button {
                    appModel.reloadCurrentModel()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Reload")
            }

            Text(appModel.selectedModelName)
                .font(.headline)
                .lineLimit(1)

            HStack(spacing: 18) {
                ForEach(appModel.steps) { step in
                    StepBadge(step: step)
                }
            }

            if let status = appModel.statusMessage {
                Text(status)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            if appModel.isReadyForQueries {
                HStack(spacing: 12) {
                    TextField("Text query", text: $appModel.queryText)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            Task { await appModel.searchCurrentQuery() }
                        }

                    Button {
                        Task { await appModel.searchCurrentQuery() }
                    } label: {
                        if appModel.isSearching {
                            ProgressView()
                        } else {
                            Image(systemName: "paperplane.fill")
                        }
                    }
                    .disabled(appModel.isSearching)
                }

                if let match = appModel.lastMatchText {
                    Text(match)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }

            if let error = appModel.errorMessage {
                Text(error)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(32)
        .frame(minWidth: 520, minHeight: 320, alignment: .topLeading)
        .fileImporter(
            isPresented: $isImporterPresented,
            allowedContentTypes: [.usdz],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case let .success(urls):
                if let url = urls.first {
                    appModel.selectUSDZ(url: url)
                }
            case let .failure(error):
                appModel.errorMessage = error.localizedDescription
            }
        }
    }
}

private struct StepBadge: View {
    let step: PipelineStepState

    var body: some View {
        VStack(spacing: 8) {
            Text("\(step.id.rawValue)")
                .font(.title2.monospacedDigit().weight(.semibold))
                .frame(width: 44, height: 44)
                .background(.thinMaterial, in: Circle())

            statusIcon
                .frame(width: 24, height: 24)

            Text(step.id.title)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(width: 78)
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch step.status {
        case .waiting:
            Image(systemName: "circle")
                .foregroundStyle(.tertiary)
        case .running:
            ProgressView()
                .controlSize(.small)
        case .done:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .failed:
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.red)
        }
    }
}

#Preview(windowStyle: .automatic) {
    ContentView()
        .environment(AppModel())
}
