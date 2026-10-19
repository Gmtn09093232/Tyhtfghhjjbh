"""
Advanced CNC G-Code Generator — Professional CAM Prototype
"""

import os
import uuid
import tempfile
import traceback
import math
from pathlib import Path
from typing import List, Dict, Any, Literal

import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    import ezdxf
    import cv2
    import pymupdf as fitz
except ImportError as e:
    raise RuntimeError(f"Missing dependency: {e.name}. Please run pip install -r requirements.txt")

# ============================================================
# DATA MODELS
# ============================================================
class GeometryModel(BaseModel):
    units: Literal["mm", "inch"] = "mm"
    entities: List[dict] = []
    bbox: dict = {}
    source_format: str = ""

class MachiningConfig(BaseModel):
    machine_type: Literal["mill", "lathe"]
    stock_diameter: float = 50.0
    stock_length: float = 150.0
    tool_diameter: float = 6.0
    feed_rate: float = 300.0
    spindle_rpm: int = 3000
    depth_of_cut: float = 2.0
    safe_z: float = 10.0

class GenerateRequest(BaseModel):
    geometry: GeometryModel
    config: MachiningConfig

# ============================================================
# GEOMETRY EXTRACTION
# ============================================================
def _bbox_from_points(points):
    xs, ys = zip(*points)
    return {"min_x": float(min(xs)), "min_y": float(min(ys)), "max_x": float(max(xs)), "max_y": float(max(ys))}

def parse_dxf(path: str) -> Dict[str, Any]:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    entities, points = [], []
    for e in msp:
        t = e.dxftype()
        if t == "LINE":
            s, en = e.dxf.start, e.dxf.end
            entities.append({"type": "line", "start": {"x": float(s.x), "y": float(s.y)}, "end": {"x": float(en.x), "y": float(en.y)}})
            points.extend([(float(s.x), float(s.y)), (float(en.x), float(en.y))])
        elif t == "LWPOLYLINE":
            pts = [(float(p[0]), float(p[1])) for p in e.get_points()]
            for i in range(len(pts) - 1):
                entities.append({"type": "line", "start": {"x": pts[i][0], "y": pts[i][1]}, "end": {"x": pts[i+1][0], "y": pts[i+1][1]}})
            points.extend(pts)
    return {"units": "mm", "entities": entities, "bbox": _bbox_from_points(points) if points else {}, "source_format": "dxf"}

def parse_raster(path: str) -> Dict[str, Any]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10)
    entities, points = [], []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if not (angle < 5 or angle > 175 or 85 < angle < 95): continue
            sx1, sy1, sx2, sy2 = float(x1), float(y1), float(x2), float(y2)
            entities.append({"type": "line", "start": {"x": sx1, "y": sy1}, "end": {"x": sx2, "y": sy2}})
            points.extend([(sx1, sy1), (sx2, sy2)])
    return {"units": "mm", "entities": entities, "bbox": _bbox_from_points(points) if points else {}, "source_format": "raster"}

# ============================================================
# ADVANCED CAM ALGORITHMS
# ============================================================
def generate_lathe_gcode(cfg: MachiningConfig):
    lines, toolpath = [], []
    d_stock = cfg.stock_diameter
    z_end = -cfg.stock_length
    d_final = d_stock - 20.0 # Simulated final diameter
    doc = cfg.depth_of_cut
    
    lines.append("%")
    lines.append("O1000 (LATHE ADVANCED)")
    lines.append("G21 G99 G40 G80 G90")
    lines.append("G54 T0101 M06")
    lines.append(f"S{cfg.spindle_rpm} M03 M08")
    lines.append(f"G00 X{d_stock + 5:.3f} Z5.0")
    
    # 3D Toolpath coords for Lathe (Z is axis, X is radius)
    toolpath.append([d_stock/2, 0, 5.0])
    
    # Facing Pass
    lines.append("(FACING)")
    lines.append(f"G00 X{d_stock:.3f} Z0.0")
    lines.append("G01 X-1.0 F0.2")
    lines.append(f"G00 X{d_stock + 2:.3f} Z2.0")
    
    # G71 Roughing Cycle
    lines.append("(ROUGHING CYCLE)")
    lines.append(f"G71 U{doc:.3f} R0.5")
    lines.append(f"G71 P100 Q200 U0.5 W0.1 F{cfg.feed_rate/1000:.3f}")
    
    lines.append("N100 G00 X0.0")
    lines.append("G01 Z0.0")
    lines.append(f"X{d_final:.3f}")
    lines.append(f"Z{z_end + 10.0:.3f}")
    lines.append(f"X{d_final + 10:.3f}")
    lines.append(f"Z{z_end:.3f}")
    lines.append(f"X{d_stock + 2:.3f}")
    lines.append("N200 G00 Z2.0")
    
    # Simulate the toolpath for 3D animation
    steps = int((d_stock - d_final) / doc) + 1
    for i in range(steps):
        cur_d = d_stock - (i * doc)
        if cur_d < d_final: cur_d = d_final
        toolpath.append([cur_d/2, 0, 2.0]) # Approach
        toolpath.append([cur_d/2, 0, z_end]) # Cut
        
    lines.append(f"G00 X{d_stock + 5:.3f} Z5.0 M09 M05 M30")
    lines.append("%")
    return "\n".join(lines), toolpath

def generate_mill_gcode(cfg: MachiningConfig):
    lines, toolpath = [], []
    d_tool = cfg.tool_diameter
    stepover = d_tool * 0.7
    doc = cfg.depth_of_cut
    z_bottom = -5.0
    x_size = cfg.stock_length
    y_size = cfg.stock_diameter
    
    lines.append("%")
    lines.append("O2000 (MILL ADVANCED)")
    lines.append("G21 G94 G17 G40 G49 G80 G90")
    lines.append("G54 T1 M06")
    lines.append(f"S{cfg.spindle_rpm} M03 M08")
    lines.append(f"G00 X0.0 Y0.0 Z{cfg.safe_z:.3f}")
    
    toolpath.append([0, 0, cfg.safe_z])
    
    # Z-Stepdowns
    z_passes = int(abs(z_bottom) / doc)
    for z_step in range(1, z_passes + 1):
        cur_z = -z_step * doc
        lines.append(f"(Z-DEPTH {cur_z:.3f})")
        
        # Zig-Zag Pocketing
        y = 0
        while y <= y_size:
            lines.append(f"G00 X0.0 Y{y:.3f} Z{cur_z + 2:.3f}")
            toolpath.append([0, y, cur_z + 2])
            
            lines.append(f"G01 Z{cur_z:.3f} F100.0")
            toolpath.append([0, y, cur_z])
            
            lines.append(f"G01 X{x_size:.3f} F{cfg.feed_rate:.3f}")
            toolpath.append([x_size, y, cur_z])
            
            y += stepover
            
    lines.append(f"G00 Z{cfg.safe_z:.3f} M09 M05 M30")
    lines.append("%")
    return "\n".join(lines), toolpath

# ============================================================
# API ROUTES
# ============================================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
UPLOAD_DIR = Path(tempfile.gettempdir()) / "cnc_adv"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    save_path.write_bytes(await file.read())
    try:
        if ext == ".dxf": geom = parse_dxf(str(save_path))
        elif ext in (".png", ".jpg", ".jpeg"): geom = parse_raster(str(save_path))
        else: raise HTTPException(400, "Unsupported format")
        return {"geometry": geom, "filename": file.filename}
    except Exception as e:
        raise HTTPException(422, str(e))

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if req.config.machine_type == "lathe":
        gcode, toolpath = generate_lathe_gcode(req.config)
    else:
        gcode, toolpath = generate_mill_gcode(req.config)
    return {"gcode": gcode, "toolpath": toolpath}

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index():
    return FileResponse(HERE / "index.html")
