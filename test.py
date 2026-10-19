import os
import sys
import tempfile
from main import (
    _bbox_from_points,
    validate_geometry,
    validate_gcode,
    MillPathGenerator,
    LathePathGenerator,
    FanucPostProcessor,
    parse_dxf,
)

def base_geom():
    return {
        "units": "mm",
        "entities": [
            {"type": "line", "start": {"x": 0.0,   "y": 0.0},  "end": {"x": 100.0, "y": 0.0}},
            {"type": "line", "start": {"x": 100.0, "y": 0.0},  "end": {"x": 100.0, "y": 60.0}},
            {"type": "line", "start": {"x": 100.0, "y": 60.0}, "end": {"x": 0.0,   "y": 60.0}},
            {"type": "line", "start": {"x": 0.0,   "y": 60.0}, "end": {"x": 0.0,   "y": 0.0}},
            {"type": "circle", "center": {"x": 50.0, "y": 30.0}, "radius": 5.0},
        ],
        "bbox": {"min_x": 0.0, "min_y": 0.0, "max_x": 100.0, "max_y": 60.0},
        "source_format": "test",
        "confidence": 1.0,
        "warnings": [],
    }

def base_cfg():
    return {
        "machine_type": "mill",
        "controller": "fanuc",
        "units": "mm",
        "work_offset": "G54",
        "stock": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 60,
                  "z_top": 0, "z_bottom": -5},
        "tool": {"number": 1, "diameter": 6.0, "type": "endmill"},
        "spindle_rpm": 3000,
        "feed_rate": 300.0,
        "feed_mode": "G94",
        "depth_of_cut": 2.0,
        "stepover": 10.0,
        "finishing_allowance": 0.2,
        "safe_z": 10.0,
        "retract_z": 5.0,
        "coolant": True,
        "operation": "facing",
        "operation_params": {},
        "max_feed": 5000.0,
        "max_rpm": 12000.0,
    }

def test_bbox():
    pts = [(0, 0), (10, 0), (10, 5), (0, 5)]
    assert _bbox_from_points(pts) == {"min_x": 0, "min_y": 0, "max_x": 10, "max_y": 5}

def test_validate_geometry_ok():
    assert validate_geometry(base_geom()) == []

def test_validate_geometry_empty():
    issues = validate_geometry({"entities": [], "bbox": {}, "confidence": 1.0})
    assert any("ERROR" in w for w in issues)

def test_mill_facing_generates_feed_moves():
    tp = MillPathGenerator(base_geom(), base_cfg()).generate()
    assert len(tp.moves) > 0
    assert any(m["type"] == "feed" for m in tp.moves)

def test_mill_contour_generates_moves():
    cfg = base_cfg(); cfg["operation"] = "contour"
    tp = MillPathGenerator(base_geom(), cfg).generate()
    feeds = [m for m in tp.moves if m["type"] == "feed"]
    assert len(feeds) == 4

def test_mill_drill_generates_drill_move():
    cfg = base_cfg(); cfg["operation"] = "drill"
    tp = MillPathGenerator(base_geom(), cfg).generate()
    drills = [m for m in tp.moves if m["type"] == "drill"]
    assert len(drills) == 1
    assert drills[0]["x"] == 50.0

def test_lathe_facing_generates_moves():
    cfg = base_cfg()
    cfg["machine_type"] = "lathe"; cfg["operation"] = "facing"
    cfg["stock"] = {"diameter": 50.0, "z_start": 0.0, "z_end": -2.0}
    tp = LathePathGenerator(base_geom(), cfg).generate()
    assert any(m["type"] == "feed" for m in tp.moves)

def test_postprocessor_header_footer():
    tp = MillPathGenerator(base_geom(), base_cfg()).generate()
    code = FanucPostProcessor(base_cfg()).generate(tp)
    assert code.startswith("%")
    assert "O0001" in code
    assert "G21" in code
    assert "M30" in code
    assert code.strip().endswith("%")

def test_postprocessor_includes_g81_for_drill():
    cfg = base_cfg(); cfg["operation"] = "drill"
    tp = MillPathGenerator(base_geom(), cfg).generate()
    code = FanucPostProcessor(cfg).generate(tp)
    assert "G81" in code
    assert "G80" in code

def test_validation_metric_inch_mismatch():
    gcode = "G20 G01 X10 F100\nM30"
    issues = validate_gcode(gcode, base_cfg())
    assert any("G20" in i["message"] for i in issues)
    assert any(i["level"] == "error" for i in issues)

def test_validation_feed_limit():
    cfg = base_cfg(); cfg["max_feed"] = 100.0
    gcode = "G01 X10 F500\nM30"
    issues = validate_gcode(gcode, cfg)
    assert any("Feed rate" in i["message"] for i in issues)

def test_validation_flags_missing_safe_end():
    gcode = "G21 G01 X10 F100"
    issues = validate_gcode(gcode, base_cfg())
    assert any("M30" in i["message"] for i in issues)

def test_dxf_parser_roundtrip():
    import ezdxf
    tmp = tempfile.NamedTemporaryFile(suffix=".dxf", delete=False)
    tmp.close()
    try:
        doc = ezdxf.new()
        msp = doc.modelspace()
        msp.add_line((0, 0), (100, 0))
        msp.add_line((100, 0), (100, 60))
        msp.add_circle((50, 30), 5)
        doc.saveas(tmp.name)
        geom = parse_dxf(tmp.name)
        assert geom["source_format"] == "dxf"
        assert len(geom["entities"]) == 3
        assert any(e["type"] == "circle" for e in geom["entities"])
    finally:
        os.unlink(tmp.name)

if __name__ == "__main__":
    tests = [
        test_bbox, test_validate_geometry_ok, test_validate_geometry_empty,
        test_mill_facing_generates_feed_moves, test_mill_contour_generates_moves,
        test_mill_drill_generates_drill_move, test_lathe_facing_generates_moves,
        test_postprocessor_header_footer, test_postprocessor_includes_g81_for_drill,
        test_validation_metric_inch_mismatch, test_validation_feed_limit,
        test_validation_flags_missing_safe_end, test_dxf_parser_roundtrip,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    sys.exit(0 if failed == 0 else 1)
