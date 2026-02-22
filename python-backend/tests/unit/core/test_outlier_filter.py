from app.core.outlier_filter import hampel_filter, double_hampel_filter


def test_no_outliers():
    data = [1.0, 1.1, 0.9, 1.0, 1.1]
    filtered, outliers = hampel_filter(data)
    assert len(outliers) == 0


def test_single_outlier():
    data = [1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0]
    filtered, outliers = hampel_filter(data, window_size=3, threshold=3.0)
    assert 3 in outliers
    assert abs(filtered[3] - 1.0) < 1.0  # Should be replaced near median


def test_double_hampel():
    data = [1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 3.0, 1.0]
    filtered, outliers = double_hampel_filter(data)
    assert len(outliers) >= 1
