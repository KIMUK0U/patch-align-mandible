//
//  ImmersiveView.swift
//  PatchSimlilarySpace
//
//  Created by 木村亘汰 on 2026/04/24.
//

import RealityKit
import SwiftUI

struct ImmersiveView: View {
    @Environment(AppModel.self) private var appModel

    var body: some View {
        RealityView { content in
            let root = Entity()
            root.name = "MandibleRoot"
            content.add(root)
            appModel.attachRealityRoot(root)
        }
        .onDisappear {
            appModel.detachRealityRoot()
        }
    }
}

#Preview(immersionStyle: .mixed) {
    ImmersiveView()
        .environment(AppModel())
}
