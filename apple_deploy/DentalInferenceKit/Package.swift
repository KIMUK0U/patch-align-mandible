// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DentalInferenceKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "DentalInferenceKit", targets: ["DentalInferenceKit"]),
    ],
    targets: [
        .target(
            name: "DentalInferenceKit",
            path: "Sources/DentalInferenceKit",
            resources: [
                // Copy the resource directory as-is so multiple CoreML
                // .mlpackage directories can coexist under SwiftPM CLI builds.
                .copy("Resources"),
            ]
        ),
    ]
)
