"""
CNC G-Code Generator — Backend (single-file, Vercel-ready).

Local run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Vercel:
    Vercel routes every request to this FastAPI app.
    `/`             → FileResponse serves index.html (same folder)
    `/api/upload`   → geometry parser
    `/api/generate` → toolpath + G-code generator
"""

import os
import uuid
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Literal

import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- dependencies ----------
try:
    import ezdxf
    from svgpathtools import svg2paths, Line, CubicBezier, QuadraticBezier, Arc
    import cv2
    import fitz  # PyMuPDF
except ImportError as e:
    raise RuntimeError(
        f"Missing dependency: {e.name}. Run: pip install -r requirements.txt"
    )

# ============================================================
# Pydantic models
# ============================================================
class GeometryModel(BaseModel):
    units: Literal["mm", "inch"] = "mm"
    entities: List[dict] = []
    bbox: dict = {}
    source_format: str = ""
    confidence: float = 1.0
    warnings: List[str] = []

class MachiningConfig(BaseModel):
    machine_type: Literal["mill", "lathe"]
    controller: Literal["fanuc"] = "fanuc"
    units: Literal["mm", "inch"] = "mm"
    work_offset: str = "G54"
    stock: dict = {}
    tool: dict = {}
    spindle_rpm: int = 3000
    feed_rate: float = 300.0
    feed_mode: Literal["G94", "G95"] = "G94"
    depth_of_cut: float = 2.0
    stepover: float = 3.0
    finishing_allowance: float = 0.2
    safe_z: float = 10.0
    retract_z: float = 5.0
    coolant: bool = True
    operation: Literal["facing", "contour", "pocket", "drill",
                       "turn_rough", "turn_finish", "groove"] = "contour"
    operation_params: dict = {}
    max_feed: float = 5000.0
    max_rpm: float = 12000.0

class GenerateRequest(BaseModel):
    geometry: GeometryModel
    config: MachiningConfig

# ============================================================
# Geometry extraction
# ============================================================
def _bbox_from_points(points):
    xs, ys = zip(*points)
    return {"min_x": min(xs), "min_y": min(ys), "max_x": max(xs), "max_y": max(ys)}

def parse_dxf(path: str) -> Dict[str, Any]:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    entities, points = [], []
    for e in msp:
        t = e.dxftype()
        if t == "LINE":
            s, en = e.dxf.start, e.dxf.end
            entities.append({"type": "line", "start": {"x": s.x, "y": s.y}, "end": {"x": en.x, "y": en.y}})
            points.extend([(s.x, s.y), (en.x, en.y)])
        elif t == "CIRCLE":
            c, r = e.dxf.center, e.dxf.radius
            entities.append({"type": "circle", "center": {"x": c.x, "y": c.y}, "radius": r})
            points.extend([(c.x - r, c.y - r), (c.x + r, c.y + r)])
        elif t == "ARC":
            c, r = e.dxf.center, e.dxf.radius
            entities.append({"type": "arc", "center": {"x": c.x, "y": c.y}, "radius": r,
                             "start_angle": e.dxf.start_angle,
                             "end_angle": e.dxf.end_angle, "clockwise": False})
            points.extend([(c.x - r, c.y - r), (c.x + r, c.y + r)])
        elif t == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            for i in range(len(pts) - 1):
                entities.append({"type": "line",
                                 "start": {"x": pts[i][0], "y": pts[i][1]},
                                 "end": {"x": pts[i + 1][0], "y": pts[i + 1][1]}})
            points.extend(pts)
        elif t == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            for i in range(len(pts) - 1):
                entities.append({"type": "line",
                                 "start": {"x": pts[i][0], "y": pts[i][1]},
                                 "end": {"x": pts[i + 1][0], "y": pts[i + 1][1]}})
            points.extend(pts)
    return {"units": "mm", "entities": entities,
            "bbox": _bbox_from_points(points) if points else {},
            "source_format": "dxf", "confidence": 1.0,
            "warnings": [] if points else ["No supported entities found in DXF modelspace."]}

def parse_svg(path: str) -> Dict[str, Any]:
    paths, _ = svg2paths(path)
    entities, points = [], []
    for path_obj in paths:
        for seg in path_obj:
            if isinstance(seg, Line):
                entities.append({"type": "line",
                                 "start": {"x": seg.start.real, "y": seg.start.imag},
                                 "end": {"x": seg.end.real, "y": seg.end.imag}})
                points.extend([(seg.start.real, seg.start.imag),
                               (seg.end.real, seg.end.imag)])
            elif isinstance(seg, (CubicBezier, QuadraticBezier)):
                n = 20
                prev = seg.point(0)
                for t in np.linspace(0, 1, n)[1:]:
                    cur = seg.point(t)
                    entities.append({"type": "line",
                                     "start": {"x": prev.real, "y": prev.imag},
                                     "end": {"x": cur.real, "y": cur.imag}})
                    points.extend([(prev.real, prev.imag), (cur.real, cur.imag)])
                    prev = cur
            elif isinstance(seg, Arc):
                entities.append({"type": "arc",
                                 "center": {"x": seg.center.real, "y": seg.center.imag},
                                 "radius": seg.radius,
                                 "start_angle": float(np.degrees(seg.theta)),
                                 "end_angle": float(np.degrees(seg.theta + seg.delta)),
                                 "clockwise": bool(seg.delta < 0)})
                points.extend([(seg.center.real - seg.radius, seg.center.imag - seg.radius),
                               (seg.center.real + seg.radius, seg.center.imag + seg.radius)])
    return {"units": "mm", "entities": entities,
            "bbox": _bbox_from_points(points) if points else {},
            "source_format": "svg", "confidence": 1.0,
            "warnings": [] if points else ["No path segments found in SVG."]}

def parse_pdf_vector(path: str) -> Dict[str, Any]:
    doc = fitz.open(path)
    entities, points = [], []
    for page in doc:
        for drawing in page.get_drawings():
            for item in drawing["items"]:
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    entities.append({"type": "line",
                                     "start": {"x": p1.x, "y": p1.y},
                                     "end": {"x": p2.x, "y": p2.y}})
                    points.extend([(p1.x, p1.y), (p2.x, p2.y)])
                elif item[0] == "c":
                    pts = item[1:]
                    prev = pts[0]
                    for cur in pts[1:]:
                        entities.append({"type": "line",
                                         "start": {"x": prev.x, "y": prev.y},
                                         "end": {"x": cur.x, "y": cur.y}})
                        points.extend([(prev.x, prev.y), (cur.x, cur.y)])
                        prev = cur
    doc.close()
    if not points:
        return {"entities": [], "bbox": {}, "source_format": "pdf", "confidence": 0.0,
                "units": "mm",
                "warnings": ["No vector paths found. Scanned PDF requires raster pipeline."]}
    return {"units": "mm", "entities": entities,
            "bbox": _bbox_from_points(points),
            "source_format": "pdf_vector", "confidence": 1.0, "warnings": []}

def parse_raster(path: str, scale_mm_per_px: float = 1.0) -> Dict[str, Any]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot read image")
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=30, maxLineGap=10)
    entities, points = [], []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            entities.append({"type": "line",
                             "start": {"x": float(x1 * scale_mm_per_px),
                                       "y": float(y1 * scale_mm_per_px)},
                             "end": {"x": float(x2 * scale_mm_per_px),
                                     "y": float(y2 * scale_mm_per_px)}})
            points.extend([(x1 * scale_mm_per_px, y1 * scale_mm_per_px),
                           (x2 * scale_mm_per_px, y2 * scale_mm_per_px)])
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=200)
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = c
            entities.append({"type": "circle",
                             "center": {"x": float(cx * scale_mm_per_px),
                                        "y": float(cy * scale_mm_per_px)},
                             "radius": float(r * scale_mm_per_px)})
            points.extend([(cx - r, cy - r), (cx + r, cy + r)])
    return {"units": "mm", "entities": entities,
            "bbox": _bbox_from_points(points) if points else {},
            "source_format": "raster",
            "confidence": 0.3 if points else 0.0,
            "warnings": ["Raster extraction is approximate. Manual correction is mandatory."]
                        if points else ["No geometry detected."]}

def extract_geometry(path: str, ext: str, scale_mm_per_px: float = 1.0) -> Dict[str, Any]:
    ext = ext.lower()
    if ext == ".dxf":  return parse_dxf(path)
    if ext == ".svg":  return parse_svg(path)
    if ext == ".pdf":  return parse_pdf_vector(path)
    if ext in (".png", ".jpg", ".jpeg"):  return parse_raster(path, scale_mm_per_px)
    raise ValueError(f"Unsupported format: {ext}")

# ============================================================
# Toolpath generation
# ============================================================
class Toolpath:
    def __init__(self):
        self.moves: List[Dict[str, Any]] = []
    def rapid(self, x, y, z):
        self.moves.append({"type": "rapid", "x": x, "y": y, "z": z})
    def feed(self, x, y, z, f=None):
        self.moves.append({"type": "feed", "x": x, "y": y, "z": z, "f": f})
    def arc(self, x, y, z, i, j, clockwise, f=None):
        self.moves.append({"type": "arc_cw" if clockwise else "arc_ccw",
                           "x": x, "y": y, "z": z, "i": i, "j": j, "f": f})

class MillPathGenerator:
    def __init__(self, geometry: Dict, config: Dict):
        self.geom = geometry
        self.cfg = config
        self.tp = Toolpath()
        self.tool_r = config["tool"].get("diameter", 6.0) / 2.0
        self.stock = config["stock"]
        self.safe_z = config.get("safe_z", 10.0)
        self.retract_z = config.get("retract_z", 5.0)
        self.feed = config.get("feed_rate", 300.0)

    def generate(self) -> Toolpath:
        op = self.cfg["operation"]
        if op == "facing":     self._facing()
        elif op == "contour":  self._contour()
        elif op == "pocket":   self._pocket()
        elif op == "drill":    self._drill()
        else:  raise ValueError(f"Unsupported mill operation: {op}")
        return self.tp

    def _safe_start(self, x, y):
        self.tp.rapid(x, y, self.safe_z)
        self.tp.rapid(x, y, self.retract_z)

    def _facing(self):
        sx, sy = self.stock["x_min"], self.stock["y_min"]
        ex, ey = self.stock["x_max"], self.stock["y_max"]
        step = self.cfg.get("stepover", 3.0)
        z = self.stock.get("z_top", 0.0)
        self._safe_start(sx, sy)
        y = sy; direction = 1
        while y <= ey:
            if direction == 1:
                self.tp.feed(sx, y, z, self.feed)
                self.tp.feed(ex, y, z, self.feed)
            else:
                self.tp.feed(ex, y, z, self.feed)
                self.tp.feed(sx, y, z, self.feed)
            y += step; direction *= -1
        self.tp.rapid(sx, sy, self.safe_z)

    def _contour(self):
        lines = [e for e in self.geom["entities"] if e["type"] == "line"]
        if not lines:
            raise ValueError("Contour requires line segments.")
        first = lines[0]
        self._safe_start(first["start"]["x"], first["start"]["y"])
        z_bottom = self.stock.get("z_bottom", -5.0)
        for seg in lines:
            self.tp.feed(seg["end"]["x"], seg["end"]["y"], z_bottom, self.feed)
        last = lines[-1]["end"]
        self.tp.rapid(last["x"], last["y"], self.safe_z)

    def _pocket(self):
        b = self.geom["bbox"]
        if not b:
            raise ValueError("Pocket requires geometry bounding box.")
        cx = (b["min_x"] + b["max_x"]) / 2
        cy = (b["min_y"] + b["max_y"]) / 2
        w = (b["max_x"] - b["min_x"]) - 2 * self.tool_r
        h = (b["max_y"] - b["min_y"]) - 2 * self.tool_r
        step = self.cfg.get("stepover", 3.0)
        z_bottom = self.stock.get("z_bottom", -5.0)
        self._safe_start(cx, cy)
        while w > 0 and h > 0:
            self.tp.feed(cx - w/2, cy - h/2, z_bottom, self.feed)
            self.tp.feed(cx + w/2, cy - h/2, z_bottom, self.feed)
            self.tp.feed(cx + w/2, cy + h/2, z_bottom, self.feed)
            self.tp.feed(cx - w/2, cy + h/2, z_bottom, self.feed)
            self.tp.feed(cx - w/2, cy - h/2, z_bottom, self.feed)
            w -= 2 * step; h -= 2 * step
        self.tp.rapid(cx, cy, self.safe_z)

    def _drill(self):
        circles = [e for e in self.geom["entities"] if e["type"] == "circle"]
        if not circles:
            raise ValueError("Drill operation requires circles.")
        z_bottom = self.stock.get("z_bottom", -5.0)
        r_plane = self.retract_z
        for c in circles:
            self.tp.rapid(c["center"]["x"], c["center"]["y"], self.safe_z)
            self.tp.moves.append({"type": "drill",
                                  "x": c["center"]["x"],
                                  "y": c["center"]["y"],
                                  "z": z_bottom, "r": r_plane})
        self.tp.rapid(circles[-1]["center"]["x"],
                      circles[-1]["center"]["y"], self.safe_z)

class LathePathGenerator:
    def __init__(self, geometry: Dict, config: Dict):
        self.geom = geometry
        self.cfg = config
        self.tp = Toolpath()
        self.safe_z = config.get("safe_z", 50.0)
        self.feed = config.get("feed_rate", 0.2)
        self.doc = config.get("depth_of_cut", 2.0)

    def generate(self) -> Toolpath:
        op = self.cfg["operation"]
        if op in ("turn_rough", "turn_finish"):  self._turn()
        elif op == "facing":  self._face()
        elif op == "groove":  self._groove()
        else:  raise ValueError(f"Unsupported lathe operation: {op}")
        return self.tp

    def _face(self):
        s = self.cfg["stock"]
        d_start = s.get("diameter", 50.0); z_start = s.get("z_start", 0.0)
        z_end = s.get("z_end", -2.0)
        self.tp.rapid(d_start, z_start, 0)
        x = d_start
        while x > 0:
            self.tp.feed(x, z_end, 0, self.feed)
            self.tp.rapid(x, z_start, 0)
            x -= 2 * self.doc
        self.tp.rapid(d_start, self.safe_z, 0)

    def _turn(self):
        s = self.cfg["stock"]
        d_start = s.get("diameter", 50.0); z_start = s.get("z_start", 0.0)
        z_end = s.get("z_end", -100.0); d_final = s.get("final_diameter", 40.0)
        self.tp.rapid(d_start + 2, z_start, 0)
        d = d_start
        while d > d_final:
            self.tp.feed(d, z_end, 0, self.feed)
            self.tp.rapid(d, z_start, 0)
            d -= 2 * self.doc
        self.tp.feed(d_final, z_end, 0, self.feed)
        self.tp.rapid(d_final + 2, self.safe_z, 0)

    def _groove(self):
        s = self.cfg["stock"]
        z_groove = s.get("z_groove", -20.0)
        width = s.get("groove_width", 3.0)
        d_start = s.get("diameter", 50.0)
        depth = s.get("groove_depth", 2.0)
        self.tp.rapid(d_start, z_groove, 0)
        self.tp.feed(d_start - depth, z_groove, 0, self.feed)
        self.tp.feed(d_start - depth, z_groove + width, 0, self.feed)
        self.tp.rapid(d_start, self.safe_z, 0)

# ============================================================
# FANUC Postprocessor
# ============================================================
class FanucPostProcessor:
    def __init__(self, config: Dict):
        self.cfg = config
        self.lines: List[str] = []
        self.line_no = 10
        self.units = config.get("units", "mm")
        self.feed_mode = config.get("feed_mode", "G94")

    def _emit(self, block: str):
        self.lines.append(f"N{self.line_no} {block}")
        self.line_no += 10

    def _header(self):
        self._emit("%")
        self._emit("O0001 (GENERATED BY CNC-GCODE-GENERATOR)")
        self._emit(f"({self.cfg['machine_type'].upper()} OPERATION: {self.cfg['operation']})")
        self._emit(f"({self.units.upper()} | {self.cfg['work_offset']} | TOOL T{self.cfg['tool'].get('number', 1)})")
        self._emit("G21" if self.units == "mm" else "G20")
        self._emit("G17 G40 G49 G80")
        self._emit(f"{self.cfg['work_offset']}")
        self._emit(f"T{self.cfg['tool'].get('number', 1)} M06")
        self._emit(f"S{self.cfg['spindle_rpm']} M03")
        if self.cfg.get("coolant"):
            self._emit("M08")
        self._emit(f"G43 H{self.cfg['tool'].get('number', 1)} Z{self.cfg['safe_z']:.3f}")
        self._emit("G99" if (self.feed_mode == "G95" and self.cfg["machine_type"] == "lathe") else "G94")

    def _footer(self):
        self._emit(f"G00 Z{self.cfg['safe_z']:.3f}")
        self._emit("M09")
        self._emit("M05")
        self._emit("G91 G28 Z0")
        self._emit("G28 X0 Y0" if self.cfg["machine_type"] == "mill" else "G28 X0")
        self._emit("G90")
        self._emit("M30")
        self._emit("%")

    def generate(self, toolpath: Toolpath) -> str:
        self._header()
        if self.cfg["machine_type"] == "lathe":
            self._emit_lathe(toolpath)
        else:
            self._emit_mill(toolpath)
        self._footer()
        return "\n".join(self.lines)

    def _emit_mill(self, tp: Toolpath):
        for m in tp.moves:
            if m["type"] == "rapid":
                self._emit(f"G00 X{m['x']:.3f} Y{m['y']:.3f} Z{m['z']:.3f}")
            elif m["type"] == "feed":
                f = m.get("f") or self.cfg["feed_rate"]
                self._emit(f"G01 X{m['x']:.3f} Y{m['y']:.3f} Z{m['z']:.3f} F{f:.3f}")
            elif m["type"] in ("arc_cw", "arc_ccw"):
                g = "G02" if m["type"] == "arc_cw" else "G03"
                f = m.get("f") or self.cfg["feed_rate"]
                self._emit(f"{g} X{m['x']:.3f} Y{m['y']:.3f} I{m['i']:.3f} J{m['j']:.3f} F{f:.3f}")
            elif m["type"] == "drill":
                self._emit(f"G81 X{m['x']:.3f} Y{m['y']:.3f} Z{m['z']:.3f} R{m['r']:.3f} F{self.cfg['feed_rate']:.3f}")
        self._emit("G80")

    def _emit_lathe(self, tp: Toolpath):
        for m in tp.moves:
            if m["type"] == "rapid":
                self._emit(f"G00 X{m['x']:.3f} Z{m['z']:.3f}")
            elif m["type"] == "feed":
                f = m.get("f") or self.cfg["feed_rate"]
                self._emit(f"G01 X{m['x']:.3f} Z{m['z']:.3f} F{f:.3f}")

# ============================================================
# Validation
# ============================================================
def validate_geometry(geom: Dict) -> List[str]:
    w = []
    if not geom.get("entities"):
        w.append("ERROR: No geometry entities were extracted.")
    if geom.get("confidence", 1.0) < 0.8:
        w.append(f"WARNING: Low confidence ({geom['confidence']:.2f}) — manual correction required.")
    if not geom.get("bbox"):
        w.append("ERROR: Bounding box could not be calculated.")
    else:
        bx = geom["bbox"]["max_x"] - geom["bbox"]["min_x"]
        by = geom["bbox"]["max_y"] - geom["bbox"]["min_y"]
        if bx <= 0 or by <= 0:
            w.append("ERROR: Invalid geometry extents (zero or negative).")
    return w

def validate_gcode(gcode: str, cfg: Dict) -> List[Dict[str, str]]:
    issues = []
    lines = gcode.splitlines()
    if not any("G40" in l and "G49" in l for l in lines):
        issues.append({"level": "warning",
                       "message": "Safe-start block (G40 G49 G80) not detected."})
    if not any("M30" in l for l in lines):
        issues.append({"level": "warning", "message": "Program end (M30) not detected."})

    max_feed = cfg.get("max_feed", 5000.0)
    for line in lines:
        if "F" in line:
            try:
                f_val = float(line.split("F")[1].split()[0])
                if f_val > max_feed:
                    issues.append({"level": "error",
                                   "message": f"Feed rate {f_val} exceeds limit {max_feed}."})
            except (IndexError, ValueError):
                pass

    max_rpm = cfg.get("max_rpm", 12000.0)
    for line in lines:
        if "S" in line and "M03" in line:
            try:
                s_val = int(float(line.split("S")[1].split()[0]))
                if s_val > max_rpm:
                    issues.append({"level": "error",
                                   "message": f"Spindle speed {s_val} RPM exceeds limit {max_rpm}."})
            except (IndexError, ValueError):
                pass

    for line in lines:
        if "G02" in line or "G03" in line:
            if "I" not in line and "J" not in line and "R" not in line:
                issues.append({"level": "error",
                               "message": f"Arc block missing I/J or R: {line.strip()}"})

    tool_d = cfg.get("tool", {}).get("diameter", 0)
    if tool_d <= 0:
        issues.append({"level": "error", "message": "Tool diameter must be > 0."})

    if cfg.get("units") == "mm" and "G20" in gcode:
        issues.append({"level": "error", "message": "Metric config but G20 (inch) found."})
    if cfg.get("units") == "inch" and "G21" in gcode:
        issues.append({"level": "error", "message": "Imperial config but G21 (mm) found."})

    issues.append({
        "level": "info",
        "message": "This G-code has NOT been machine-tested. Perform independent "
                   "backplotting / simulation and a controlled prove-out before machining."
    })
    return issues

# ============================================================
# FastAPI application
# ============================================================
app = FastAPI(title="CNC G-Code Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

HERE = Path(__file__).resolve().parent
UPLOAD_DIR = Path(tempfile.gettempdir()) / "cnc_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload_drawing(file: UploadFile = File(...),
                         scale_mm_per_px: float = Form(1.0)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    allowed = {".dxf", ".svg", ".pdf", ".png", ".jpg", ".jpeg"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {sorted(allowed)}")
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    save_path.write_bytes(await file.read())
    try:
        geom = extract_geometry(str(save_path), ext, scale_mm_per_px)
    except Exception as e:
        raise HTTPException(422, f"Geometry extraction failed: {e}")
    geom["warnings"] = validate_geometry(geom) + geom.get("warnings", [])
    return {"geometry": geom, "filename": file.filename, "format": ext}

@app.post("/api/generate")
async def generate_gcode(req: GenerateRequest):
    geom = req.geometry.model_dump()
    cfg = req.config.model_dump()
    if cfg["machine_type"] == "mill":
        gen = MillPathGenerator(geom, cfg)
    else:
        gen = LathePathGenerator(geom, cfg)
    try:
        tp = gen.generate()
    except Exception as e:
        raise HTTPException(422, f"Toolpath generation failed: {e}")
    post = FanucPostProcessor(cfg)
    gcode = post.generate(tp)
    issues = validate_gcode(gcode, cfg)
    return {"gcode": gcode, "issues": issues, "line_count": len(gcode.splitlines())}

# ---- Serve index.html (root of project — no public/ folder) ----
@app.get("/", include_in_schema=False)
async def index():
    candidate = HERE / "index.html"
    if candidate.exists():
        return FileResponse(candidate)
    return JSONResponse(
        {"error": "index.html not found",
         "expected_at": str(candidate)},
        status_code=500,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
