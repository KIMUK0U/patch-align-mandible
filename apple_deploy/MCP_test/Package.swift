// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DentalPatchMCP",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "DentalPatchMCPServer", targets: ["DentalPatchMCPServer"]),
    ],
    dependencies: [
        .package(path: "../DentalInferenceKit"),
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.11.0"),
    ],
    targets: [
        .executableTarget(
            name: "DentalPatchMCPServer",
            dependencies: [
                .product(name: "DentalInferenceKit", package: "DentalInferenceKit"),
                .product(name: "MCP", package: "swift-sdk"),
            ]
        ),
    ]
)
