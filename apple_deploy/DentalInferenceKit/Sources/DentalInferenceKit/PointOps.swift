/// FPS + KNN point cloud grouping — pure Swift, no external dependencies.
/// Mirrors PatchAlign3D PatchedGroup logic for on-device CPU preprocessing.
import Accelerate
import Foundation

public enum PointOps {

    // MARK: - Types

    public struct GroupResult {
        /// (G * M * 3) relative neighbourhood coords, row-major
        public let neighborhood: [Float]
        /// (G * 3) absolute patch centre coords, row-major
        public let centers: [Float]
        /// (G * M) absolute indices into original xyz
        public let patchIdx: [Int32]
        public let G: Int
        public let M: Int
    }

    // MARK: - Public API

    /// Group `xyz` into `numGroup` patches of `groupSize` points via FPS + KNN.
    ///
    /// - Parameter xyz: Flat float array, N*3 row-major [x0,y0,z0, x1,y1,z1, ...]
    /// - Returns: GroupResult with neighbourhood coords relative to each centre
    public static func group(
        xyz: [Float],
        numGroup G: Int,
        groupSize M: Int
    ) -> GroupResult {
        let N = xyz.count / 3
        precondition(N >= G, "Point cloud has \(N) points but \(G) groups requested")

        let centerIdx = fps(xyz: xyz, N: N, G: G)      // [G] indices

        var centers = [Float](repeating: 0, count: G * 3)
        for g in 0..<G {
            let ci = Int(centerIdx[g])
            centers[g * 3 + 0] = xyz[ci * 3 + 0]
            centers[g * 3 + 1] = xyz[ci * 3 + 1]
            centers[g * 3 + 2] = xyz[ci * 3 + 2]
        }

        let idx = knn(xyz: xyz, N: N, centers: centers, G: G, k: M)  // [G*M]

        var neighborhood = [Float](repeating: 0, count: G * M * 3)
        for g in 0..<G {
            for m in 0..<M {
                let ni = Int(idx[g * M + m])
                neighborhood[(g * M + m) * 3 + 0] = xyz[ni * 3 + 0] - centers[g * 3 + 0]
                neighborhood[(g * M + m) * 3 + 1] = xyz[ni * 3 + 1] - centers[g * 3 + 1]
                neighborhood[(g * M + m) * 3 + 2] = xyz[ni * 3 + 2] - centers[g * 3 + 2]
            }
        }

        return GroupResult(neighborhood: neighborhood, centers: centers,
                           patchIdx: idx, G: G, M: M)
    }

    /// Downsample `xyz` to `targetCount` points using Furthest Point Sampling.
    ///
    /// - Parameter xyz: Flat Float array, N*3 row-major
    /// - Parameter targetCount: Number of output points (must be ≤ N)
    /// - Returns: Flat Float array, targetCount*3, selected points in original scale
    public static func voxelFPSDownsample(xyz: [Float], targetCount: Int) throws -> [Float] {
        let N = xyz.count / 3
        guard N >= targetCount else {
            throw PointOpsError.insufficientPoints(have: N, need: targetCount)
        }
        let indices = fps(xyz: xyz, N: N, G: targetCount)
        var result = [Float](repeating: 0, count: targetCount * 3)
        for i in 0..<targetCount {
            let src = Int(indices[i]) * 3
            result[i*3+0] = xyz[src+0]
            result[i*3+1] = xyz[src+1]
            result[i*3+2] = xyz[src+2]
        }
        return result
    }

    /// Centre and scale xyz to unit sphere, in-place.
    public static func normalise(xyz: inout [Float]) {
        let N = xyz.count / 3
        var mx: Float = 0, my: Float = 0, mz: Float = 0
        for i in 0..<N {
            mx += xyz[i * 3]; my += xyz[i * 3 + 1]; mz += xyz[i * 3 + 2]
        }
        let fn = Float(N)
        mx /= fn; my /= fn; mz /= fn
        for i in 0..<N {
            xyz[i * 3] -= mx; xyz[i * 3 + 1] -= my; xyz[i * 3 + 2] -= mz
        }
        var maxAbs: Float = 0
        for v in xyz { maxAbs = max(maxAbs, abs(v)) }
        var scale = 1.0 / (maxAbs + 1e-8)
        vDSP_vsmul(xyz, 1, &scale, &xyz, 1, vDSP_Length(xyz.count))
    }

    // MARK: - Errors

    public enum PointOpsError: Error {
        case insufficientPoints(have: Int, need: Int)
    }

    // MARK: - Private

    /// Furthest Point Sampling — O(N * G)
    private static func fps(xyz: [Float], N: Int, G: Int) -> [Int32] {
        var selected = [Int32](repeating: 0, count: G)
        var dist     = [Float](repeating: .infinity, count: N)
        var cur      = 0

        for i in 0..<G {
            selected[i] = Int32(cur)
            let cx = xyz[cur * 3], cy = xyz[cur * 3 + 1], cz = xyz[cur * 3 + 2]
            var maxDist: Float = -.infinity
            var nextCur        = 0
            for n in 0..<N {
                let dx = xyz[n * 3] - cx
                let dy = xyz[n * 3 + 1] - cy
                let dz = xyz[n * 3 + 2] - cz
                let d  = dx*dx + dy*dy + dz*dz
                if d < dist[n] { dist[n] = d }
                if dist[n] > maxDist { maxDist = dist[n]; nextCur = n }
            }
            cur = nextCur
        }
        return selected
    }

    /// Brute-force K-Nearest Neighbours — O(G * N)
    private static func knn(
        xyz: [Float], N: Int,
        centers: [Float], G: Int, k: Int
    ) -> [Int32] {
        var result = [Int32](repeating: 0, count: G * k)
        var dist2  = [Float](repeating: 0, count: N)

        for g in 0..<G {
            let cx = centers[g * 3], cy = centers[g * 3 + 1], cz = centers[g * 3 + 2]
            for n in 0..<N {
                let dx = xyz[n * 3] - cx
                let dy = xyz[n * 3 + 1] - cy
                let dz = xyz[n * 3 + 2] - cz
                dist2[n] = dx*dx + dy*dy + dz*dz
            }
            // Partial sort: pick k smallest
            var order = Array(0..<N)
            order.sort { dist2[$0] < dist2[$1] }
            for m in 0..<k {
                result[g * k + m] = Int32(order[m])
            }
        }
        return result
    }
}
