from app.core.vessel_mesher import generate_vessel_mesh


def test_mesh_generation():
    centerline = [(0, 0, i * 1.0) for i in range(20)]
    diameters = [3.0] * 20
    mesh = generate_vessel_mesh(centerline, diameters, segments_per_ring=8)
    assert len(mesh["positions"]) > 0
    assert len(mesh["indices"]) > 0
    assert mesh["num_vertices"] == 20 * 8
