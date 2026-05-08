import CoreML
import Foundation

enum ResourceLocator {
    private static let resourceSubdirectory = "DentalInferenceResources"

    static func url(forResource name: String, withExtension ext: String) -> URL? {
        // アプリBundle直下
        if let direct = Bundle.main.url(forResource: name, withExtension: ext) {
            return direct
        }

        // アプリBundle/DentalInferenceResources/
        if let nested = Bundle.main.url(
            forResource: name,
            withExtension: ext,
            subdirectory: resourceSubdirectory
        ) {
            return nested
        }

        return nil
    }

    static func coreMLModelURL(named name: String) throws -> URL {
        if let compiledURL = url(forResource: name, withExtension: "mlmodelc") {
            return compiledURL
        }

        if let packageURL = url(forResource: name, withExtension: "mlpackage") {
            return try MLModel.compileModel(at: packageURL)
        }

        throw ResourceError.missingCoreMLModel(name)
    }

    enum ResourceError: LocalizedError {
        case missingCoreMLModel(String)

        var errorDescription: String? {
            switch self {
            case .missingCoreMLModel(let name):
                return "\(name).mlmodelc or \(name).mlpackage not found in app bundle"
            }
        }
    }
}
