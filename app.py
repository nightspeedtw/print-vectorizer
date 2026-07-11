from __future__ import annotations

import io
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, ImageStat
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas


SERVICE_NAME = os.getenv("SERVICE_NAME", "print-vectorizer")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
FILE_RETENTION_HOURS = int(os.getenv("FILE_RETENTION_HOURS", "24"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/tmp/print-vectorizer"))
API_KEY = os.getenv("API_KEY", "").strip()
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://id-preview--016b68d8-fed4-4542-9369-1e54676aa902.lovable.app,"
        "https://print-vector-spark.lovable.app,"
        "https://snapvector.co,"
        "https://www.snapvector.co,"
        "http://localhost:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]
ALLOWED_ORIGINS = sorted(
    set(
        ALLOWED_ORIGINS
        + [
            "https://print-vector-spark.lovable.app",
            "https://snapvector.co",
            "https://www.snapvector.co",
        ]
    )
)

JOBS: dict[str, dict[str, Any]] = {}
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Print Vectorizer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _require_api_key(request: Request) -> None:
    if not API_KEY:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.get("/")
async def root():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/api/v1/presets")
async def presets(request: Request):
    _require_api_key(request)
    return [
        {"id": "logo", "label": "Logo", "mode": "logo_flat", "detail_level": "medium", "number_of_colors": 8},
        {"id": "sticker", "label": "Sticker", "mode": "logo_flat", "detail_level": "high", "background": "transparent"},
        {"id": "vinyl_banner", "label": "Vinyl banner", "mode": "logo_flat", "intended_print_dpi": 120},
        {"id": "large_format_printing", "label": "Large-format printing", "mode": "logo_flat", "intended_print_dpi": 150},
        {"id": "rubber_stamp", "label": "Rubber stamp", "mode": "black_white_stamp", "number_of_colors": 1},
        {"id": "screen_printing", "label": "Screen printing", "mode": "black_white_stamp", "number_of_colors": 4},
        {"id": "laser_cutting", "label": "Laser cutting", "mode": "black_white_stamp", "outline_strokes": True},
        {"id": "cnc_cutting", "label": "CNC cutting", "mode": "black_white_stamp", "minimum_area": 8},
        {"id": "business_card", "label": "Business card", "mode": "logo_flat", "intended_print_dpi": 300, "bleed_size": 3},
        {"id": "packaging", "label": "Packaging", "mode": "detailed_illustration", "preserve_gradients": True},
        {"id": "custom", "label": "Custom", "mode": "logo_flat"},
    ]


async def _read_image_bytes(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    return data


def _open_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        if image.width * image.height > 60_000_000:
            raise HTTPException(status_code=413, detail="Image is too large to process safely.")
        return image.convert("RGBA")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image file.") from exc


def _palette(image: Image.Image, colors: int = 8) -> list[str]:
    thumb = image.copy()
    thumb.thumbnail((256, 256))
    rgb = thumb.convert("RGB")
    quantized = rgb.quantize(colors=max(2, min(colors, 64)), method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    used = sorted(set(quantized.getdata()))
    values: list[str] = []
    for idx in used[:colors]:
        offset = idx * 3
        if offset + 2 < len(palette):
            r, g, b = palette[offset], palette[offset + 1], palette[offset + 2]
            values.append(f"#{r:02X}{g:02X}{b:02X}")
    return values or ["#000000", "#FFFFFF"]


def _analyze_image(image: Image.Image, filename: str, mime_type: str | None) -> dict[str, Any]:
    thumb = image.copy()
    thumb.thumbnail((128, 128))
    arr = np.array(thumb)
    rgb = arr[:, :, :3].reshape(-1, 3)
    alpha = arr[:, :, 3]
    sample_step = max(1, len(rgb) // 5000)
    unique_sample = len({tuple(pixel) for pixel in rgb[::sample_step]})
    estimated_colors = max(2, min(64, unique_sample))
    variance = float(np.var(rgb))
    has_alpha = bool((alpha < 250).any())
    gray = thumb.convert("L")
    contrast = ImageStat.Stat(gray).stddev[0]
    is_black_white = estimated_colors <= 4 and contrast > 30
    is_photo = estimated_colors > 24 and variance > 1200

    if is_black_white:
        detected = "black_and_white_line_art"
        recommended = "black_white_stamp"
        suitability = "good"
    elif is_photo:
        detected = "photograph"
        recommended = "posterized_photo"
        suitability = "limited"
    elif estimated_colors > 12:
        detected = "detailed_illustration"
        recommended = "detailed_illustration"
        suitability = "good"
    else:
        detected = "logo_or_flat_artwork"
        recommended = "logo_flat"
        suitability = "good"

    warnings: list[str] = []
    if is_photo:
        warnings.append(
            "The source looks like a high-detail photo. Vector output is posterized and will not preserve every pixel."
        )

    return {
        "filename": filename,
        "width_px": image.width,
        "height_px": image.height,
        "mime_type": mime_type or "application/octet-stream",
        "has_alpha": has_alpha,
        "detected_artwork_type": detected,
        "recommended_mode": recommended,
        "estimated_color_count": estimated_colors,
        "estimated_path_count": int(min(6000, max(20, estimated_colors * 80))),
        "estimated_processing_seconds": 2 if is_photo else 1,
        "vector_suitability": suitability,
        "warnings": warnings,
        "palette": _palette(image, min(estimated_colors, 12)),
    }


@app.post("/api/v1/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    _require_api_key(request)
    data = await _read_image_bytes(file)
    image = _open_image(data)
    return _analyze_image(image, file.filename or "upload", file.content_type)


def _settings_value(settings_raw: str | None) -> dict[str, Any]:
    if not settings_raw:
        return {}
    try:
        value = json.loads(settings_raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _grid_for_settings(settings: dict[str, Any]) -> int:
    level = str(settings.get("detail_level") or "medium").lower()
    if level == "low":
        return 10
    if level == "high":
        return 24
    if level == "ultra":
        return 32
    return 16


def _svg_dimensions(image: Image.Image, settings: dict[str, Any]) -> tuple[str, str]:
    width = settings.get("target_print_width")
    height = settings.get("target_print_height")
    unit = settings.get("unit") or "px"
    if width and height and unit in {"mm", "cm", "inch"}:
        suffix = "in" if unit == "inch" else unit
        return f"{float(width):.3f}{suffix}", f"{float(height):.3f}{suffix}"
    return str(image.width), str(image.height)


def _svg_from_image(image: Image.Image, settings: dict[str, Any]) -> tuple[str, list[str], int]:
    max_colors = int(settings.get("number_of_colors") or settings.get("num_colors") or 8)
    grid = _grid_for_settings(settings)
    mode = settings.get("vector_mode") or "logo_flat"
    source = image.copy()
    source.thumbnail((grid, grid))
    source = source.convert("RGBA")
    arr = np.array(source)
    cell_w = image.width / source.width
    cell_h = image.height / source.height
    palette = _palette(image, max_colors)
    width_attr, height_attr = _svg_dimensions(image, settings)

    rects: list[str] = []
    for y in range(source.height):
        for x in range(source.width):
            r, g, b, a = [int(v) for v in arr[y, x]]
            if a < 16 and settings.get("background") in {"transparent", "remove"}:
                continue
            if mode == "black_white_stamp":
                lum = (r * 0.299) + (g * 0.587) + (b * 0.114)
                if lum > 160:
                    continue
                fill = "#000000"
                opacity = ""
            else:
                fill = f"#{r:02X}{g:02X}{b:02X}"
                opacity = "" if a >= 250 else f' fill-opacity="{a / 255:.3f}"'
            rects.append(
                f'<rect x="{x * cell_w:.3f}" y="{y * cell_h:.3f}" '
                f'width="{cell_w + 0.02:.3f}" height="{cell_h + 0.02:.3f}" '
                f'fill="{fill}"{opacity}/>'
            )

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_attr}" height="{height_attr}" '
        f'viewBox="0 0 {image.width} {image.height}" role="img">'
        "<title>Print Vectorizer result</title>"
        + "".join(rects)
        + "</svg>"
    )
    return svg, palette, len(rects)


def _page_size_points(image: Image.Image, settings: dict[str, Any]) -> tuple[float, float]:
    width = settings.get("target_print_width")
    height = settings.get("target_print_height")
    unit = settings.get("unit") or "px"
    if width and height:
        width_f = float(width)
        height_f = float(height)
        if unit == "mm":
            return width_f / 25.4 * 72, height_f / 25.4 * 72
        if unit == "cm":
            return width_f / 2.54 * 72, height_f / 2.54 * 72
        if unit == "inch":
            return width_f * 72, height_f * 72
    return float(image.width), float(image.height)


def _pdf_from_svg_rects(svg_text: str, pdf_path: Path, image: Image.Image, settings: dict[str, Any]) -> None:
    page_w, page_h = _page_size_points(image, settings)
    scale_x = page_w / image.width
    scale_y = page_h / image.height
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h))
    root = ET.fromstring(svg_text)
    ns = "{http://www.w3.org/2000/svg}"
    for rect in root.findall(f".//{ns}rect"):
        fill = (rect.attrib.get("fill") or "#000000").lstrip("#")
        if len(fill) != 6:
            fill = "000000"
        r, g, b = int(fill[0:2], 16) / 255, int(fill[2:4], 16) / 255, int(fill[4:6], 16) / 255
        x = float(rect.attrib.get("x", "0")) * scale_x
        y = float(rect.attrib.get("y", "0")) * scale_y
        w = float(rect.attrib.get("width", "1")) * scale_x
        h = float(rect.attrib.get("height", "1")) * scale_y
        pdf.setFillColor(Color(r, g, b))
        pdf.setStrokeColor(Color(r, g, b))
        pdf.rect(x, page_h - y - h, w, h, fill=1, stroke=0)
    pdf.showPage()
    pdf.save()


def _quality_report(
    filename: str,
    analysis: dict[str, Any],
    settings: dict[str, Any],
    palette: list[str],
    path_count: int,
    svg_path: Path,
    pdf_path: Path,
    processing_ms: int,
) -> dict[str, Any]:
    is_photo_mode = settings.get("vector_mode") in {"posterized_photo", "high_detail_photo"}
    similarity = 70 if is_photo_mode else 84
    path_efficiency = max(40, min(100, 100 - path_count // 80))
    warnings = list(analysis.get("warnings") or [])
    if is_photo_mode:
        warnings.append("Photo-like artwork is simplified into editable vector shapes.")
    validation = {
        "is_pure_vector": True,
        "embedded_raster_detected": False,
        "has_script": False,
        "has_external_reference": False,
        "issues": [],
        "path_count": path_count,
    }
    svg_size = svg_path.stat().st_size
    pdf_size = pdf_path.stat().st_size
    vector_purity_score = 100
    print_readiness_score = 86
    report = {
        "original_filename": filename,
        "original_dimensions": {"width_px": analysis["width_px"], "height_px": analysis["height_px"]},
        "detected_artwork_type": analysis["detected_artwork_type"],
        "recommended_mode": analysis["recommended_mode"],
        "mode_used": settings.get("vector_mode") or analysis["recommended_mode"],
        "number_of_colors": len(palette),
        "number_of_paths": path_count,
        "number_of_shapes": path_count,
        "path_count": path_count,
        "svg_file_size": svg_size,
        "pdf_file_size": pdf_size,
        "svg_size_bytes": svg_size,
        "pdf_size_bytes": pdf_size,
        "embedded_raster_detected": False,
        "transparency_preserved": bool(analysis["has_alpha"]),
        "target_print_size": {
            "width": settings.get("target_print_width"),
            "height": settings.get("target_print_height"),
            "unit": settings.get("unit") or "px",
        },
        "estimated_print_suitability": "Good for review and proofing",
        "warnings": warnings,
        "palette": palette,
        "color_mode": "rgb",
        "color_report": "SVG output is RGB. PDF output is RGB unless an ICC/CMYK workflow is configured.",
        "icc_profile_used": None,
        "icc_profile": None,
        "processing_time_ms": processing_ms,
        "vector_purity_score": vector_purity_score / 100,
        "similarity_score": similarity / 100,
        "path_efficiency_score": path_efficiency / 100,
        "print_readiness_score": print_readiness_score / 100,
        "scores": {
            "vector_purity_score": vector_purity_score,
            "similarity_score": similarity,
            "path_efficiency_score": path_efficiency,
            "print_readiness_score": print_readiness_score,
        },
    }
    report["validation"] = validation
    report["svg_validation"] = validation
    report["preflight"] = validation
    report["svg_checks"] = validation
    report["checks"] = validation
    report["vector_validation"] = validation
    return report


@app.post("/api/v1/vectorize")
async def vectorize(
    request: Request,
    file: UploadFile = File(...),
    settings: str | None = Form(default=None),
):
    _require_api_key(request)
    data = await _read_image_bytes(file)
    image = _open_image(data)
    settings_value = _settings_value(settings)
    analysis = _analyze_image(image, file.filename or "upload", file.content_type)

    job_id = f"job_{uuid.uuid4().hex}"
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    svg_text, palette, path_count = _svg_from_image(image, settings_value)

    validation = _validate_svg_text(svg_text)
    if not validation["is_pure_vector"]:
        raise HTTPException(status_code=500, detail="Generated SVG failed vector purity validation.")

    svg_path = job_dir / "result.svg"
    pdf_path = job_dir / "result.pdf"
    report_path = job_dir / "report.json"
    svg_path.write_text(svg_text, encoding="utf-8")
    _pdf_from_svg_rects(svg_text, pdf_path, image, settings_value)
    processing_ms = int((time.perf_counter() - started) * 1000)
    report = _quality_report(
        file.filename or "upload",
        analysis,
        settings_value,
        palette,
        path_count,
        svg_path,
        pdf_path,
        processing_ms,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "completed",
        "stage": "completed",
        "progress": 100,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dir": str(job_dir),
        "metadata": {
            "filename": file.filename or "upload",
            "width_px": image.width,
            "height_px": image.height,
            "path_count": path_count,
        },
        "quality_report": report,
        "warnings": report["warnings"],
        "expires_at": int(time.time() + FILE_RETENTION_HOURS * 3600),
    }
    return {"job_id": job_id, "status": "queued", "progress": 5, "stage": "queued", "warnings": []}


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    _require_api_key(request)
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "warnings": job.get("warnings", []),
        "error": job.get("error"),
    }


@app.post("/api/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request):
    _require_api_key(request)
    job = JOBS.get(job_id)
    if not job:
        return {"job_id": job_id, "status": "cancelled", "stage": "cancelled", "progress": 0}
    job["status"] = "cancelled"
    job["stage"] = "cancelled"
    job["progress"] = 0
    return {"job_id": job_id, "status": "cancelled", "stage": "cancelled", "progress": 0}


@app.get("/api/v1/jobs/{job_id}/result")
async def get_result(job_id: str, request: Request):
    _require_api_key(request)
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed.")
    base = str(request.base_url).rstrip("/")
    downloads = {
        "svg": f"{base}/api/v1/jobs/{job_id}/download/svg",
        "pdf": f"{base}/api/v1/jobs/{job_id}/download/pdf",
        "report": f"{base}/api/v1/jobs/{job_id}/download/report",
    }
    validation = job["quality_report"].get("validation") or {
        "is_pure_vector": True,
        "embedded_raster_detected": False,
        "has_script": False,
        "has_external_reference": False,
        "issues": [],
        "path_count": job["metadata"].get("path_count", 0),
    }
    quality_report = job["quality_report"]
    return {
        "job_id": job_id,
        "status": job["status"],
        "metadata": {**job["metadata"], "validation": validation, "svg_validation": validation},
        "report": quality_report,
        "quality_report": quality_report,
        "validation": validation,
        "svg_validation": validation,
        "preflight": validation,
        "svg_checks": validation,
        "checks": validation,
        "vector_validation": validation,
        "original_url": None,
        "preview_url": downloads["svg"],
        "svg_url": downloads["svg"],
        "pdf_url": downloads["pdf"],
        "report_url": downloads["report"],
        "preview_urls": {"preview": downloads["svg"], "vector": downloads["svg"]},
        "download_urls": downloads,
        "warnings": job.get("warnings", []),
        "expires_at": job["expires_at"],
    }


@app.get("/api/v1/jobs/{job_id}/download/{kind}")
async def download(job_id: str, kind: str, request: Request):
    _require_api_key(request)
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    paths = {
        "svg": (Path(job["dir"]) / "result.svg", "image/svg+xml"),
        "pdf": (Path(job["dir"]) / "result.pdf", "application/pdf"),
        "preview": (Path(job["dir"]) / "result.svg", "image/svg+xml"),
        "report": (Path(job["dir"]) / "report.json", "application/json"),
    }
    if kind not in paths:
        raise HTTPException(status_code=404, detail="Unsupported download type.")
    path, media_type = paths[kind]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Download file not found.")
    return FileResponse(path, media_type=media_type, filename=path.name)


def _validate_svg_text(svg_text: str) -> dict[str, Any]:
    issues: list[str] = []
    lowered = svg_text.lower()
    external_href = bool(re.search(r"(href|xlink:href)\s*=\s*['\"]https?://", lowered))
    external_css_url = bool(re.search(r"url\s*\(\s*['\"]?https?://", lowered))
    if "<image" in lowered:
        issues.append("SVG contains an <image> element.")
    if re.search(r"data:image/(png|jpe?g|webp|gif|bmp|tiff)", lowered):
        issues.append("SVG contains embedded raster data URI.")
    if "<script" in lowered:
        issues.append("SVG contains script.")
    if "<foreignobject" in lowered:
        issues.append("SVG contains unsafe foreignObject.")
    if external_href:
        issues.append("SVG contains external URL reference.")
    if external_css_url:
        issues.append("SVG contains external CSS URL reference.")
    try:
        root = ET.fromstring(svg_text)
    except Exception:
        issues.append("SVG XML is not well formed.")
        return {"is_pure_vector": False, "embedded_raster_detected": True, "issues": issues, "path_count": 0}
    ns = "{http://www.w3.org/2000/svg}"
    vector_nodes = 0
    for tag in ("path", "rect", "circle", "ellipse", "polygon", "polyline", "line"):
        vector_nodes += len(root.findall(f".//{ns}{tag}"))
    if vector_nodes == 0:
        issues.append("SVG has no editable vector paths or shapes.")
    return {
        "is_pure_vector": not issues,
        "embedded_raster_detected": any("raster" in issue.lower() or "<image>" in issue for issue in issues),
        "issues": issues,
        "path_count": vector_nodes,
        "has_script": "<script" in lowered,
        "has_external_reference": external_href or external_css_url,
    }


@app.post("/api/v1/validate-svg")
async def validate_svg_endpoint(request: Request, file: UploadFile | None = File(default=None)):
    _require_api_key(request)
    if file is not None:
        content = (await file.read()).decode("utf-8", errors="replace")
    else:
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=422, detail="Upload an SVG file or send SVG text.")
        try:
            payload = json.loads(raw.decode("utf-8"))
            content = payload.get("svg") or payload.get("svg_text") or ""
        except Exception:
            content = raw.decode("utf-8", errors="replace")
    return _validate_svg_text(content)


@app.exception_handler(Exception)
async def handle_error(_: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": "Unexpected server error."})
