from app.services.export_service import export_csv, export_json


def test_export_csv_with_data():
    qca = {0: {"mld_mm": 1.5, "diameter_stenosis_pct": 45.2, "method": "gaussian"}}
    rws = [{"start_frame": 0, "end_frame": 10, "mld_rws_pct": 8.5, "interpretation": "intermediate"}]
    csv_str = export_csv("test-session", qca, rws)
    assert "For Research Use Only" in csv_str
    assert "1.5" in csv_str
    assert "8.5" in csv_str


def test_export_csv_empty():
    csv_str = export_csv("test", {}, [])
    assert "For Research Use Only" in csv_str


def test_export_json_with_data():
    qca = {0: {"mld_mm": 1.5}}
    rws = [{"mld_rws_pct": 8.5}]
    json_str = export_json("test-session", qca, rws)
    assert "disclaimer" in json_str
    assert "1.5" in json_str


def test_export_json_empty():
    json_str = export_json("test", {}, [])
    assert "disclaimer" in json_str
