"""
CNC G-Code Generator — Modern Backend (Render-Ready)
"""

import os
import uuid
import tempfile
import traceback
import re
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
# PYDANTIC MODELS
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
    stock: dict = {}
    tool: dict = {}
    spindle_rpm: int = 3000
    feed_rate: float = 300.0
    depth_of_cut: float = 2.0
    safe_z: float = 10.0
    coolant: bool = True
    operation: str = "contour"

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
        elif t == "CIRCLE":
            c, r = e.dxf.center, e.dxf.radius
            entities.append({"type": "circle", "center": {"x": float(c.x), "y": float(c.y)}, "radius": float(r)})
            points.extend([(float(c.x) - float(r), float(c.y) - float(r)), (float(c.x) + float(r), float(c.y) + float(r))])
    return {"units": "mm", "entities": entities, "bbox": _bbox_from_points(points) if points else {},
            "source_format": "dxf", "confidence": 1.0, "warnings": [] if points else ["No entities found."]}

def parse_raster(path: str, scale: float = 1.0) -> Dict[str, Any]:
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError("Cannot read image.")
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10)
        entities, points = [], []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Filter out diagonal arrows/dimension lines
                if not (angle < 5 or angle > 175 or 85 < angle < 95): continue
                
                sx1, sy1, sx2, sy2 = float(x1*scale), float(y1*scale), float(x2*scale), float(y2*scale)
                entities.append({"type": "line", "start": {"x": sx1, "y": sy1}, "end": {"x": sx2, "y": sy2}})
                points.extend([(sx1, sy1), (sx2, sy2)])
                
        return {"units": "mm", "entities": entities, "bbox": _bbox_from_points(points) if points else {},
                "source_format": "raster", "confidence": 0.6, 
                "warnings": ["Raster extraction is approximate. Use DXF for precise geometry."]}
    except Exception as e:
        raise ValueError(f"Image parsing failed: {str(e)}")

# ============================================================
# TOOLPATH GENERATION
# ============================================================
def generate_gcode_and_path(geom: Dict, cfg: Dict):
    lines = []
    toolpath_coords = [] # Store X, Y, Z for 3D frontend viewer
    
    # Header
    lines.append("%")
    lines.append("O1001 (MODERN CNC GENERATOR)")
    lines.append(f"({cfg['machine_type'].upper()} - {cfg['operation'].upper()})")
    lines.append("G21 G17 G40 G49 G80 G90")
    lines.append("G54")
    lines.append(f"T{cfg['tool'].get('number', 1)} M06")
    lines.append(f"S{cfg['spindle_rpm']} M03")
    if cfg.get("coolant"): lines.append("M08")
    lines.append(f"G00 Z{cfg['safe_z']:.3f}")

    if cfg["machine_type"] == "mill":
        b = geom["bbox"]
        cx = (b["min_x"] + b["max_x"]) / 2
        cy = (b["min_y"] + b["max_y"]) / 2
        z_depth = cfg["stock"].get("z_bottom", -5.0)
        
        # Simple Pocket / Contour Logic
        lines.append(f"G00 X{cx:.3f} Y{cy:.3f}")
        lines.append(f"G01 Z{z_depth:.3f} F{cfg['feed_rate']/2:.3f}")
        
        # Dummy toolpath square to show 3D visualization
        step = 10.0
        for i in range(3):
            offset = i * step
            lines.append(f"G01 X{cx - 20 + offset:.3f} Y{cy - 20 + offset:.3f} F{cfg['feed_rate']:.3f}")
            lines.append(f"G01 X{cx + 20 - offset:.3f} Y{cy - 20 + offset:.3f}")
            lines.append(f"G01 X{cx + 20 - offset:.3f} Y{cy + 20 - offset:.3f}")
            lines.append(f"G01 X{cx - 20 + offset:.3f} Y{cy + 20 - offset:.3f}")
            lines.append(f"G01 X{cx - 20 + offset:.3f} Y{cy - 20 + offset:.3f}")
            toolpath_coords.append([cx, cy, z_depth]) # simplified

        lines.append(f"G00 Z{cfg['safe_z']:.3f}")

    elif cfg["machine_type"] == "lathe":
        stock_dia = cfg["stock"].get("diameter", 50.0)
        z_end = cfg["stock"].get("z_end", -150.0)
        final_dia = cfg["stock"].get("final_diameter", 30.0)
        
        lines.append(f"G00 X{stock_dia + 2:.3f} Z2.0")
        
        # Simulate G71 Roughing Cycle
        lines.append(f"(ROUGHING CYCLE)")
        lines.append(f"G71 U{cfg['depth_of_cut']:.3f} R0.5")
        lines.append(f"G71 P10 Q20 U0.5 W0.1 F{cfg['feed_rate']/1000:.3f}")
        lines.append(f"N10 G00 X{final_dia:.3f}")
        lines.append(f"G01 Z0.0")
        lines.append(f"Z{z_end:.3f}")
        lines.append(f"X{stock_dia:.3f}")
        lines.append(f"N20 G00 Z2.0")
        
        # Generate coords for 3D line
        toolpath_coords = [
            [stock_dia, 0, 2], [stock_dia, 0, z_end], [final_dia, 0, z_end]
        ]
        
    # Footer
    lines.append(f"G00 Z{cfg['safe_z']:.3f}")
    lines.append("M09")
    lines.append("M05")
    lines.append("G28 U0 W0" if cfg["machine_type"] == "lathe" else "G28 X0 Y0 Z0")
    lines.append("M30")
    lines.append("%")
    
    return "\n".join(lines), toolpath_coords

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Modern CNC Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"Server Error: {str(exc)}"})

UPLOAD_DIR = Path(tempfile.gettempdir()) / "cnc_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent

@app.post("/api/upload")
async def upload_drawing(file: UploadFile = File(...), scale: float = Form(1.0)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    save_path.write_bytes(await file.read())
    
    try:
        if ext == ".dxf": geom = parse_dxf(str(save_path))
        elif ext in (".png", ".jpg", ".jpeg"): geom = parse_raster(str(save_path), scale)
        else: raise HTTPException(400, "Unsupported format. Use DXF or Image.")
    except Exception as e:
        raise HTTPException(422, f"Extraction failed: {e}")
        
    return {"geometry": geom, "filename": file.filename}

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    geom = req.geometry.model_dump()
    cfg = req.config.model_dump()
    
    try:
        gcode, toolpath = generate_gcode_and_path(geom, cfg)
    except Exception as e:
        raise HTTPException(422, f"G-Code generation failed: {e}")
        
    return {"gcode": gcode, "toolpath": toolpath}

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index():
    return FileResponse(HERE / "index.html")
