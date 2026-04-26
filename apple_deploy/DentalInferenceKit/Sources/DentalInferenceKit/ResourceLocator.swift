import CoreML
import Foundation

enum ResourceLocator {
    static func url(forResource name: String, withExtension ext: String) -> URL? {
        let bundle = Bundle.module
        if let direct = bundle.url(forResource: name, withExtension: ext) {
            return direct
        }
        if let nested = bundle.url(
            forResource: name,
            withExtension: ext,
            subdirectory: "Resources"
        ) {
            return nested
        }

        guard let candidate = bundle.resourceURL?
            .appendingPathComponent("Resources", isDirectory: true)
            .appendingPathComponent("\(name).\(ext)")
        else {
            return nil
        }
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
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
                return "\(name).mlmodelc or \(name).mlpackage not found in bundle"
            }
        }
    }
}
