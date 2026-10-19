"""
ForgeCAM AI — Advanced CAM Architecture
Integrates Geometry, Tooling, Material, Strategy, Validation, and Postprocessing.
"""

import os
import uuid
import tempfile
import traceback
import math
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional

import numpy as np
import cv2
import pymupdf as fitz
import ezdxf
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# 1. DATA MODELS (The "Elements")
# ============================================================
class Tool(BaseModel):
    id: str
    type: Literal["endmill", "turning_insert", "drill"]
    diameter: float
    corner_radius: float = 0.0 # Critical for Lathe

class Material(BaseModel):
    name: str
    cutting_speed: float # SFM (Surface Feet per Minute) or m/min
    feed_factor: float

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
    operation: Literal["contour", "rough_turn", "finish_turn"]
    tool: Tool
    material: Material
    spindle_limit: int = 8000
    feed_rate: float = 300.0
    depth_of_cut: float = 2.0
    safe_z: float = 15.0
    stock_dia: float = 50.0
    stock_length: float = 150.0

class GenerateRequest(BaseModel):
    geometry: GeometryModel
    config: OperationConfig

# ============================================================
# 2. GEOMETRY ENGINE (The "What")
# ============================================================
def _generate_bbox(points):
    if not points: return {"min_x":0, "min_y":0, "max_x":100, "max_y":50}
    xs, ys = zip(*points)
    return {"min_x": float(min(xs)), "min_y": float(min(ys)), "max_x": float(max(xs)), "max_y": float(max(ys))}

def parse_dxf(path: str) -> GeometryModel:
    try:
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
    except Exception as e:
        raise ValueError(f"DXF Parse Error: {e}")

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
# 3. CAM & VALIDATION ENGINE (The Interactions)
# ============================================================
class ToolpathValidator:
    """Checks interactions between Tool, Geometry, and Machine limits."""
    @staticmethod
    def validate(geom: GeometryModel, cfg: OperationConfig) -> List[str]:
        warnings = []
        
        # Interaction 1: Tool Size vs Geometry Internal Corners
        if cfg.machine_type == "mill" and cfg.tool.diameter > 0:
            bbox_w = geom.bbox.get("max_x", 0) - geom.bbox.get("min_x", 0)
            if cfg.tool.diameter > bbox_w * 0.8:
                warnings.append(f"WARNING: Tool Ø{cfg.tool.diameter}mm is very large compared to the feature width ({bbox_w:.1f}mm). Risk of tool rub.")
                
        # Interaction 2: Spindle Limit vs Material Cutting Speed
        # RPM = (Cutting Speed * 1000) / (PI * Tool Diameter)
        if cfg.tool.diameter > 0:
            calc_rpm = (cfg.material.cutting_speed * 1000) / (math.pi * cfg.tool.diameter)
            if calc_rpm > cfg.spindle_limit:
                warnings.append(f"WARNING: Calculated RPM ({calc_rpm:.0f}) exceeds machine limit ({cfg.spindle_limit}). Spindle will be capped.")
                
        return warnings

def extract_lathe_profile(entities):
    lines = [e for e in entities if e.type == "line"]
    if not lines: return []
    try:
        sorted_lines = sorted(lines, key=lambda e: min(e.start["x"], e.end["x"]))
        profile = []
        for line in sorted_lines:
            p1 = line.start if line.start["x"] < line.end["x"] else line.end
            p2 = line.end if line.start["x"] < line.end["x"] else line.start
            if not profile: profile.append({"z": p1["x"], "r": abs(p1["y"])})
            profile.append({"z": p2["x"], "r": abs(p2["y"])})
        clean = []
        for p in profile:
            if not clean or (abs(p["z"] - clean[-1]["z"]) > 0.1 or abs(p["r"] - clean[-1]["r"]) > 0.1):
                clean.append(p)
        if len(clean) > 1: return clean
    except: pass
    return []

class LatheStrategy:
    def __init__(self, geom: GeometryModel, cfg: OperationConfig):
        self.geom, self.cfg = geom, cfg
        self.warnings = ToolpathValidator.validate(geom, cfg)

    def generate(self):
        profile = extract_lathe_profile(self.geom.entities)
        if not profile:
            max_y = self.geom.bbox.get("max_y", self.cfg.stock_dia/2)
            max_x = self.geom.bbox.get("max_x", self.cfg.stock_length)
            profile = [{"z": 0, "r": max_y}, {"z": -max_x, "r": max_y}]
            
        moves = [{"type": "rapid", "x": self.cfg.stock_dia/2, "y": 0, "z": 5.0}]
        
        # Roughing Passes (Interaction: Depth of cut & Stock Dia)
        target_r = min(p["r"] for p in profile)
        curr_r = self.cfg.stock_dia / 2
        z_end = profile[-1]["z"]
        
        while curr_r > target_r:
            curr_r -= self.cfg.depth_of_cut
            if curr_r < target_r: curr_r = target_r
            moves.append({"type": "feed", "x": curr_r, "y": 0, "z": z_end, "f": self.cfg.feed_rate})
            moves.append({"type": "rapid", "x": curr_r, "y": 0, "z": 2.0})
            
        # Finish Pass (Follows actual geometry)
        for p in profile:
            moves.append({"type": "feed", "x": p["r"], "y": 0, "z": p["z"], "f": self.cfg.feed_rate})
            
        moves.append({"type": "rapid", "x": self.cfg.stock_dia/2, "y": 0, "z": self.cfg.safe_z})
        final_profile = [{"z": 0, "r": 0}] + profile + [{"z": profile[-1]["z"], "r": 0}]
        return moves, final_profile, self.warnings

class MillStrategy:
    def __init__(self, geom: GeometryModel, cfg: OperationConfig):
        self.geom, self.cfg = geom, cfg
        self.warnings = ToolpathValidator.validate(geom, cfg)

    def generate(self):
        lines = [e for e in self.geom.entities if e.type == "line"]
        if not lines:
            lines = [GeometryEntity(type="line", start={"x":0,"y":0}, end={"x":100,"y":0}),
                     GeometryEntity(type="line", start={"x":100,"y":0}, end={"x":100,"y":50}),
                     GeometryEntity(type="line", start={"x":100,"y":50}, end={"x":0,"y":50}),
                     GeometryEntity(type="line", start={"x":0,"y":50}, end={"x":0,"y":0})]
            
        moves = []
        start = lines[0].start
        moves.append({"type": "rapid", "x": start["x"], "y": start["y"], "z": self.cfg.safe_z})
        moves.append({"type": "feed", "x": start["x"], "y": start["y"], "z": -self.cfg.depth_of_cut, "f": 100})
        
        for line in lines:
            moves.append({"type": "feed", "x": line.end["x"], "y": line.end["y"], "z": -self.cfg.depth_of_cut, "f": self.cfg.feed_rate})
            
        moves.append({"type": "rapid", "x": line.end["x"], "y": line.end["y"], "z": self.cfg.safe_z})
        path_points = [[start["x"], start["y"]]] + [[l.end["x"], l.end["y"]] for l in lines]
        return moves, path_points, self.warnings

# ============================================================
# 4. POSTPROCESSOR ENGINE (The "Machine Dialect")
# ============================================================
class Fanuc0iTF:
    def __init__(self, moves, cfg, profile):
        self.moves, self.cfg, self.profile = moves, cfg, profile
        self.lines, self.line_no = [], 10

    def _emit(self, code): self.lines.append(f"N{self.line_no} {code}"); self.line_no += 10

    def build(self):
        self._emit("%"); self._emit("O2000 (LATHE PROFILE)"); self._emit("G21 G40 G80 G99 G90")
        self._emit("G54 T0101 M06"); self._emit(f"S{self.cfg.spindle_limit} M03"); self._emit("M08")
        self._emit(f"G00 X{self.cfg.stock_dia + 5:.3f} Z5.0")
        self._emit(f"G71 U{self.cfg.depth_of_cut:.3f} R0.5")
        self._emit(f"G71 P10 Q20 U0.5 W0.1 F{self.cfg.feed_rate/1000:.3f}")
        self._emit("N10 G00 X0.0"); self._emit("G01 Z0.0")
        for p in self.profile:
            if p["r"] > 0: self._emit(f"G01 X{p['r']*2:.3f} Z{p['z']:.3f}")
        self._emit(f"X{self.cfg.stock_dia:.3f}"); self._emit("N20 G00 Z2.0")
        self._emit(f"G70 P10 Q20 F{self.cfg.feed_rate/2000:.3f}") 
        self._emit(f"G00 X{self.cfg.stock_dia + 10:.3f} Z{self.cfg.safe_z:.3f}")
        self._emit("M09 M05"); self._emit("G28 U0 W0"); self._emit("M30"); self._emit("%")
        return "\n".join(self.lines)

class Fanuc0iMF:
    def __init__(self, moves, cfg):
        self.moves, self.cfg = moves, cfg
        self.lines, self.line_no = [], 10

    def _emit(self, code): self.lines.append(f"N{self.line_no} {code}"); self.line_no += 10

    def build(self):
        self._emit("%"); self._emit("O1000 (MILL CONTOUR)"); self._emit("G21 G17 G40 G49 G80 G90")
        self._emit("G54 T1 M06"); self._emit(f"S{self.cfg.spindle_limit} M03"); self._emit("M08")
        
        for m in self.moves:
            if m["type"] == "rapid": self._emit(f"G00 X{m['x']:.3f} Y{m['y']:.3f} Z{m['z']:.3f}")
            elif m["type"] == "feed": self._emit(f"G01 X{m['x']:.3f} Y{m['y']:.3f} Z{m['z']:.3f} F{m['f']:.1f}")
                
        self._emit(f"G00 Z{self.cfg.safe_z:.3f}")
        self._emit("M09 M05"); self._emit("G91 G28 Z0"); self._emit("G28 X0 Y0"); self._emit("G90 M30"); self._emit("%")
        return "\n".join(self.lines)

# ============================================================
# 5. API
# ============================================================
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
        
        if not geom.entities:
            geom.entities = [GeometryEntity(type="line", start={"x": 0, "y": 0}, end={"x": 100, "y": 0}), GeometryEntity(type="line", start={"x": 100, "y": 0}, end={"x": 100, "y": 50})]
            geom.bbox = {"min_x": 0, "min_y": 0, "max_x": 100, "max_y": 50}
            
        return {"geometry": geom.dict(), "filename": file.filename}
    except Exception as e:
        raise HTTPException(422, f"Geometry Error: {str(e)}")

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    geom, cfg = req.geometry, req.config
    
    try:
        if cfg.machine_type == "mill":
            moves, path_data, warnings = MillStrategy(geom, cfg).generate()
            post = Fanuc0iMF(moves, cfg)
            profile = path_data
        else:
            moves, profile, warnings = LatheStrategy(geom, cfg).generate()
            post = Fanuc0iTF(moves, cfg, profile)
            
    except Exception as e:
        raise HTTPException(422, f"CAM Engine Error: {e}")
        
    gcode = post.build()
    toolpath_coords = [[m["x"], m["y"], m["z"]] for m in moves]
        
    return {
        "gcode": gcode, 
        "warnings": warnings,
        "toolpath": toolpath_coords,
        "profile": profile
    }

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index():
    return FileResponse(HERE / "index.html")
