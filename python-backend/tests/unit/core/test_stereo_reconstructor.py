import numpy as np
from app.core.stereo_reconstructor import reconstruct_3d_centerline, compute_angular_separation, ProjectionParams


def test_angular_separation():
    assert abs(compute_angular_separation(0, 30) - 30) < 0.1
    assert abs(compute_angular_separation(350, 10) - 20) < 0.1


def test_reconstruction_produces_points():
    cl1 = [(float(x), 256.0) for x in range(200, 300, 5)]
    cl2 = [(float(x), 256.0) for x in range(200, 300, 5)]
    params1 = ProjectionParams(
        primary_angle=0, secondary_angle=0, sid=1000, sod=1000,
        pixel_spacing=0.3, image_size=(512, 512),
    )
    params2 = ProjectionParams(
        primary_angle=30, secondary_angle=0, sid=1000, sod=1000,
        pixel_spacing=0.3, image_size=(512, 512),
    )
    recon = reconstruct_3d_centerline(cl1, cl2, params1, params2)
    assert len(recon.points_3d) > 0
    assert all(len(p) == 3 for p in recon.points_3d)
    # Arc-length fractions should be monotonically increasing [0..1]
    assert len(recon.t1_fractions) == len(recon.points_3d)
    assert len(recon.t2_fractions) == len(recon.points_3d)
    assert recon.t1_fractions[0] >= 0
    assert recon.t1_fractions[-1] <= 1.0
