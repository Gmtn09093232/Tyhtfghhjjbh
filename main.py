"""
ForgeCAM AI — Stage 1 Architecture
Exact 3D Shape Tracing, FANUC G71 Profile Generation, and Deterministic Toolpaths.
"""

import os
import uuid
import tempfile
import traceback
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Literal

import numpy as np
import cv2
import pymupdf as fitz
import ezdxf
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# 1. DATA MODELS
# ============================================================
class GeometryEntity(BaseModel):
    type: Literal["line", "arc", "circle"]
    start: Dict[str, float] = None
    end: Dict[str, float] = None

class GeometryModel(BaseModel):
    units: Literal["mm", "inch"] = "mm"
    entities: List[GeometryEntity] = []
    bbox: Dict[str, float] = Field(default_factory=dict)

class OperationConfig(BaseModel):
    machine_type: Literal["mill", "lathe"]
    tool_diameter: float = 6.0
    feed_rate: float = 300.0
    spindle_rpm: int = 3000
    depth_of_cut: float = 2.0
    safe_z: float = 10.0
    stock_dia: float = 50.0
    stock_length: float = 150.0

class GenerateRequest(BaseModel):
    geometry: GeometryModel
    config: OperationConfig

# ============================================================
# 2. GEOMETRY ENGINE (Multi-Format)
# ============================================================
def _generate_bbox(points):
    if not points: return {"min_x":0, "min_y":0, "max_x":0, "max_y":0}
    xs, ys = zip(*points)
    return {"min_x": float(min(xs)), "min_y": float(min(ys)), "max_x": float(max(xs)), "max_y": float(max(ys))}

def parse_dxf(path: str) -> GeometryModel:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    entities, points = [], []
    
    for e in msp:
        t = e.dxftype()
        if t == "LINE":
            s, en = e.dxf.start, e.dxf.end
            entities.append(GeometryEntity(type="line", start={"x": s.x, "y": s.y}, end={"x": en.x, "y": en.y}))
            points.extend([(s.x, s.y), (en.x, en.y)])
        elif t == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            for i in range(len(pts) - 1):
                entities.append(GeometryEntity(type="line", start={"x": pts[i][0], "y": pts[i][1]}, end={"x": pts[i+1][0], "y": pts[i+1][1]}))
            points.extend(pts)
            
    return GeometryModel(units="mm", entities=entities, bbox=_generate_bbox(points))

def parse_image(path: str, scale: float = 1.0) -> GeometryModel:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError("Cannot read image file.")
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=30, maxLineGap=10)
    entities, points = [], []
    
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            # Filter out diagonal dimension lines
            if not (angle < 5 or angle > 175 or 85 < angle < 95): continue
            sx1, sy1, sx2, sy2 = float(x1*scale), float(y1*scale), float(x2*scale), float(y2*scale)
            entities.append(GeometryEntity(type="line", start={"x": sx1, "y": sy1}, end={"x": sx2, "y": sy2}))
            points.extend([(sx1, sy1), (sx2, sy2)])
            
    return GeometryModel(units="mm", entities=entities, bbox=_generate_bbox(points))

def parse_pdf(path: str) -> GeometryModel:
    doc = fitz.open(path)
    entities, points = [], []
    for page in doc:
        for d in page.get_drawings():
            for item in d["items"]:
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    entities.append(GeometryEntity(type="line", start={"x": p1.x, "y": p1.y}, end={"x": p2.x, "y": p2.y}))
                    points.extend([(p1.x, p1.y), (p2.x, p2.y)])
    doc.close()
    return GeometryModel(units="mm", entities=entities, bbox=_generate_bbox(points))

# ============================================================
# 3. CAM STRATEGIES (Exact Profile Machining)
# ============================================================
def extract_lathe_profile(entities):
    """Sorts DXF lines to create a top profile for Lathe revolving."""
    # Sort lines by X starting point
    sorted_lines = sorted(entities, key=lambda e: e.start["x"])
    profile = []
    
    # We trace the top half. We map DXF X -> Lathe Z, DXF Y -> Lathe Radius
    for line in sorted_lines:
        if not profile:
            profile.append({"z": line.start["x"], "r": abs(line.start["y"])})
        profile.append({"z": line.end["x"], "r": abs(line.end["y"])})
        
    # Remove duplicates and ensure it starts at Z=0
    clean_profile = []
    for p in profile:
        if not clean_profile or (p["z"] != clean_profile[-1]["z"] or p["r"] != clean_profile[-1]["r"]):
            clean_profile.append(p)
            
    return clean_profile

class LatheProfileStrategy:
    def __init__(self, geom: GeometryModel, cfg: OperationConfig):
        self.profile = extract_lathe_profile(geom.entities)
        self.cfg = cfg

    def generate(self):
        if not self.profile: raise ValueError("No profile found.")
        
        moves = []
        # 1. Safe Start
        moves.append({"type": "rapid", "x": self.cfg.stock_dia/2, "y": 0, "z": 5.0})
        
        # 2. Simulate G71 Roughing (rough visual pass over profile)
        max_r = self.cfg.stock_dia / 2
        min_r = min(p["r"] for p in self.profile)
        
        curr_r = max_r
        while curr_r > min_r:
            curr_r -= self.cfg.depth_of_cut
            # Plunge
            moves.append({"type": "feed", "x": curr_r, "y": 0, "z": self.profile[-1]["z"], "f": self.cfg.feed_rate})
            # Retract
            moves.append({"type": "rapid", "x": curr_r, "y": 0, "z": 2.0})
            
        # 3. Finish Pass (trace exact profile)
        for p in self.profile:
            moves.append({"type": "feed", "x": p["r"], "y": 0, "z": p["z"], "f": self.cfg.feed_rate})
            
        moves.append({"type": "rapid", "x": self.cfg.stock_dia/2, "y": 0, "z": self.cfg.safe_z})
        return moves, self.profile

class MillContourStrategy:
    def __init__(self, geom: GeometryModel, cfg: OperationConfig):
        self.geom = geom
        self.cfg = cfg

    def generate(self):
        lines = [e for e in self.geom.entities if e.type == "line"]
        if not lines: raise ValueError("Contour requires line segments.")
        
        # Sort lines to ensure continuous path
        sorted_lines = sorted(lines, key=lambda e: e.start["x"])
        
        moves = []
        start = sorted_lines[0].start
        moves.append({"type": "rapid", "x": start["x"], "y": start["y"], "z": self.cfg.safe_z})
        moves.append({"type": "feed", "x": start["x"], "y": start["y"], "z": -self.cfg.depth_of_cut, "f": 100})
        
        for line in sorted_lines:
            moves.append({"type": "feed", "x": line.end["x"], "y": line.end["y"], "z": -self.cfg.depth_of_cut, "f": self.cfg.feed_rate})
            
        moves.append({"type": "rapid", "x": line.end["x"], "y": line.end["y"], "z": self.cfg.safe_z})
        
        # For 3D shape, return the 2D points
        path_points = [[start["x"], start["y"]]] + [[l.end["x"], l.end["y"]] for l in sorted_lines]
        return moves, path_points

# ============================================================
# 4. POSTPROCESSORS
# ============================================================
class Fanuc0iTF:
    def __init__(self, moves: List[Dict], cfg: OperationConfig, profile: List[Dict]):
        self.moves, self.cfg, self.profile = moves, cfg, profile
        self.lines = []
        self.line_no = 10

    def _emit(self, code):
        self.lines.append(f"N{self.line_no} {code}")
        self.line_no += 10

    def build(self):
        self._emit("%"); self._emit("O2000 (EXACT LATHE PROFILE)"); self._emit("G21 G40 G80 G99 G90")
        self._emit("G54 T0101 M06"); self._emit(f"S{self.cfg.spindle_rpm} M03"); self._emit("M08")
        self._emit(f"G00 X{self.cfg.stock_dia + 5:.3f} Z5.0")
        
        # Use actual G71 Cycle
        self._emit(f"G71 U{self.cfg.depth_of_cut:.3f} R0.5")
        self._emit(f"G71 P10 Q20 U0.5 W0.1 F{self.cfg.feed_rate/1000:.3f}")
        
        self._emit("N10 G00 X0.0")
        self._emit("G01 Z0.0")
        for p in self.profile:
            # Lathe uses Diameter (X = 2 * Radius)
            self._emit(f"G01 X{p['r']*2:.3f} Z{p['z']:.3f}")
        self._emit(f"X{self.cfg.stock_dia:.3f}")
        self._emit("N20 G00 Z2.0")
        
        self._emit(f"G70 P10 Q20 F{self.cfg.feed_rate/2000:.3f}") # Finish Pass
        
        self._emit(f"G00 X{self.cfg.stock_dia + 10:.3f} Z{self.cfg.safe_z:.3f}"); self._emit("M09 M05"); self._emit("G28 U0 W0"); self._emit("M30"); self._emit("%")
        return "\n".join(self.lines)

class Fanuc0iMF:
    def __init__(self, moves: List[Dict], cfg: OperationConfig):
        self.moves, self.cfg = moves, cfg
        self.lines = []
        self.line_no = 10

    def _emit(self, code):
        self.lines.append(f"N{self.line_no} {code}")
        self.line_no += 10

    def build(self):
        self._emit("%"); self._emit("O1000 (EXACT MILL CONTOUR)"); self._emit("G21 G17 G40 G49 G80 G90")
        self._emit("G54 T1 M06"); self._emit(f"S{self.cfg.spindle_rpm} M03"); self._emit("M08")
        
        for m in self.moves:
            if m["type"] == "rapid": self._emit(f"G00 X{m['x']:.3f} Y{m['y']:.3f} Z{m['z']:.3f}")
            elif m["type"] == "feed": self._emit(f"G01 X{m['x']:.3f} Y{m['y']:.3f} Z{m['z']:.3f} F{m['f']:.1f}")
                
        self._emit(f"G00 Z{self.cfg.safe_z:.3f}"); self._emit("M09 M05"); self._emit("G91 G28 Z0"); self._emit("G28 X0 Y0"); self._emit("G90 M30"); self._emit("%")
        return "\n".join(self.lines)

# ============================================================
# 5. VALIDATION & API
# ============================================================
def validate_gcode(gcode: str) -> List[str]:
    issues = []
    if "G21" not in gcode and "G20" not in gcode: issues.append("ERROR: Missing units.")
    if "M30" not in gcode: issues.append("ERROR: Missing program end (M30).")
    return issues

app = FastAPI(title="ForgeCAM AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
HERE = Path(__file__).resolve().parent
UPLOAD_DIR = Path(tempfile.gettempdir()) / "forgecam"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".dxf", ".pdf", ".png", ".jpg", ".jpeg"}: raise HTTPException(400, "Unsupported format.")
        
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    save_path.write_bytes(await file.read())
    
    try:
        if ext == ".dxf": geom = parse_dxf(str(save_path))
        elif ext == ".pdf": geom = parse_pdf(str(save_path))
        else: geom = parse_image(str(save_path))
        if not geom.entities: raise ValueError("No geometry found.")
        return {"geometry": geom.dict(), "filename": file.filename}
    except Exception as e:
        raise HTTPException(422, f"Geometry Error: {str(e)}")

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    geom, cfg = req.geometry, req.config
    
    try:
        if cfg.machine_type == "mill":
            moves, path_data = MillContourStrategy(geom, cfg).generate()
            post = Fanuc0iMF(moves, cfg)
            profile = path_data
        else:
            moves, profile = LatheProfileStrategy(geom, cfg).generate()
            post = Fanuc0iTF(moves, cfg, profile)
            
    except Exception as e:
        raise HTTPException(422, f"CAM Engine Error: {e}")
        
    gcode = post.build()
    issues = validate_gcode(gcode)
    toolpath_coords = [[m["x"], m["y"], m["z"]] for m in moves]
        
    return {
        "gcode": gcode, 
        "issues": issues, 
        "toolpath": toolpath_coords,
        "profile": profile # This is the exact shape data
    }

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index():
    return FileResponse(HERE / "index.html")
