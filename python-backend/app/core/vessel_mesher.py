"""3D tube mesh generation from vessel centerline and diameters.

Generates a mesh suitable for Three.js visualization with QFR heatmap colors.
Uses Bishop frame (parallel transport) to prevent mesh twisting on curved vessels.
"""
import numpy as np
import math


def generate_vessel_mesh(
    centerline_3d: list[tuple[float, float, float]],
    diameters_mm: list[float],
    qfr_profile: list[float] | None = None,
    segments_per_ring: int = 16,
) -> dict:
    """Generate a tubular mesh from 3D centerline and diameters.

    Uses Bishop frame (parallel transport) for twist-free tube generation.
    Returns flat arrays suitable for Three.js BufferGeometry.
    """
    if len(centerline_3d) < 2 or len(diameters_mm) < 2:
        return {"positions": [], "normals": [], "indices": [], "colors": []}

    pts = np.array(centerline_3d, dtype=np.float64)
    n_points = len(pts)
    n_segs = segments_per_ring

    # Compute tangents for all points
    tangents = np.zeros((n_points, 3))
    for i in range(n_points):
        if i == 0:
            tangent = pts[1] - pts[0]
        elif i == n_points - 1:
            tangent = pts[-1] - pts[-2]
        else:
            tangent = pts[i + 1] - pts[i - 1]
        tangent_len = np.linalg.norm(tangent)
        tangents[i] = tangent / tangent_len if tangent_len > 1e-10 else np.array([0, 0, 1])

    # Initialize frame with arbitrary up-vector method
    normal_frames = np.zeros((n_points, 3))
    binormal_frames = np.zeros((n_points, 3))

    t0 = tangents[0]
    up = np.array([1, 0, 0]) if abs(t0[0]) < 0.9 else np.array([0, 1, 0])
    binormal = np.cross(t0, up)
    binormal = binormal / np.linalg.norm(binormal)
    normal = np.cross(binormal, t0)
    normal = normal / np.linalg.norm(normal)
    normal_frames[0] = normal
    binormal_frames[0] = binormal

    # Propagate frame using parallel transport (Bishop frame)
    for i in range(1, n_points):
        normal_frames[i], binormal_frames[i] = _parallel_transport(
            normal_frames[i - 1], tangents[i - 1], tangents[i]
        )

    # Generate ring vertices
    positions = []
    colors = []

    for i in range(n_points):
        radius = diameters_mm[i] / 2 if i < len(diameters_mm) else diameters_mm[-1] / 2
        radius = max(radius, 0.1)

        nrm = normal_frames[i]
        bnm = binormal_frames[i]

        qfr_val = qfr_profile[i] if qfr_profile and i < len(qfr_profile) else 1.0
        r, g, b = _qfr_to_color(qfr_val)

        for j in range(n_segs):
            angle = 2 * math.pi * j / n_segs
            offset = radius * (math.cos(angle) * nrm + math.sin(angle) * bnm)
            vertex = pts[i] + offset
            positions.extend(vertex.tolist())
            colors.extend([r, g, b])

    # Center mesh (translate centroid to origin for OrbitControls)
    n_verts = n_points * n_segs
    pos_arr = np.array(positions).reshape(n_verts, 3)
    centroid = np.mean(pos_arr, axis=0)
    pos_arr -= centroid
    positions = pos_arr.flatten().tolist()

    # Generate triangle indices
    indices = []
    for i in range(n_points - 1):
        for j in range(n_segs):
            curr = i * n_segs + j
            next_j = i * n_segs + (j + 1) % n_segs
            curr_next = (i + 1) * n_segs + j
            next_j_next = (i + 1) * n_segs + (j + 1) % n_segs

            indices.extend([curr, next_j, curr_next])
            indices.extend([next_j, next_j_next, curr_next])

    # Compute face-weighted vertex normals
    normals = _calculate_normals(pos_arr, indices, n_verts)

    return {
        "positions": positions,
        "normals": normals,
        "indices": indices,
        "colors": colors,
        "num_vertices": n_verts,
        "num_triangles": (n_points - 1) * n_segs * 2,
    }


def _parallel_transport(
    prev_normal: np.ndarray,
    prev_tangent: np.ndarray,
    curr_tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transport coordinate frame with minimal rotation (Bishop frame).

    Uses Rodrigues' rotation formula to rotate the previous normal
    by the angle between consecutive tangents.
    """
    axis = np.cross(prev_tangent, curr_tangent)
    axis_len = np.linalg.norm(axis)

    if axis_len < 1e-6:
        # Tangents are parallel — frame doesn't change
        binormal = np.cross(curr_tangent, prev_normal)
        bn_len = np.linalg.norm(binormal)
        if bn_len > 1e-10:
            binormal = binormal / bn_len
        return prev_normal.copy(), binormal

    axis = axis / axis_len

    dot_product = np.clip(np.dot(prev_tangent, curr_tangent), -1.0, 1.0)
    angle = np.arccos(dot_product)

    # Rodrigues' rotation formula
    k = axis
    v = prev_normal
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    k_cross_v = np.cross(k, v)
    k_dot_v = np.dot(k, v)

    curr_normal = v * cos_a + k_cross_v * sin_a + k * k_dot_v * (1 - cos_a)
    curr_normal = curr_normal / np.linalg.norm(curr_normal)

    curr_binormal = np.cross(curr_tangent, curr_normal)
    curr_binormal = curr_binormal / np.linalg.norm(curr_binormal)

    return curr_normal, curr_binormal


def _calculate_normals(
    vertices: np.ndarray, indices: list[int], n_vertices: int,
) -> list[float]:
    """Compute face-weighted vertex normals."""
    normals = np.zeros((n_vertices, 3))
    n_tris = len(indices) // 3

    for t in range(n_tris):
        i0, i1, i2 = indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        normals[i0] += face_normal
        normals[i1] += face_normal
        normals[i2] += face_normal

    # Normalize
    for i in range(n_vertices):
        norm = np.linalg.norm(normals[i])
        if norm > 0:
            normals[i] /= norm

    return normals.flatten().tolist()


def _qfr_to_color(qfr: float) -> tuple[float, float, float]:
    """Map QFR value to RGB color (0-1 range).

    1.00 -> green (#22c55e)
    0.90 -> lime (#84cc16)
    0.80 -> yellow (#eab308)
    0.75 -> orange (#f97316)
    0.70 -> red (#ef4444)
    0.60 -> dark red (#dc2626)
    """
    stops = [
        (1.00, (0.133, 0.773, 0.369)),  # green
        (0.90, (0.518, 0.800, 0.086)),  # lime
        (0.80, (0.918, 0.702, 0.031)),  # yellow
        (0.75, (0.976, 0.451, 0.086)),  # orange
        (0.70, (0.937, 0.267, 0.267)),  # red
        (0.60, (0.863, 0.149, 0.149)),  # dark red
    ]

    qfr = max(0.6, min(1.0, qfr))

    for i in range(len(stops) - 1):
        v1, c1 = stops[i]
        v2, c2 = stops[i + 1]
        if qfr >= v2:
            t = (qfr - v2) / (v1 - v2) if v1 != v2 else 0
            return (
                c2[0] + t * (c1[0] - c2[0]),
                c2[1] + t * (c1[1] - c2[1]),
                c2[2] + t * (c1[2] - c2[2]),
            )

    return stops[-1][1]
