from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
from typing import Any

import httpx

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
import pyembroidery
from fastapi.middleware.cors import CORSMiddleware
from openai import AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("khayt-ai")

APP_NAME = "khayt-ai"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
AI_PROVIDER = os.getenv("AI_PROVIDER", "cloudflare").strip().lower()
CF_TEXT_MODEL = os.getenv("CLOUDFLARE_TEXT_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
CF_VISION_MODEL = os.getenv("CLOUDFLARE_VISION_MODEL", "@cf/meta/llama-3.2-11b-vision-instruct")
CF_AUDIO_MODEL = os.getenv("CLOUDFLARE_AUDIO_MODEL", "@cf/openai/whisper")
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_AUDIO_TYPES = {
    "audio/webm", "audio/ogg", "audio/wav", "audio/mpeg",
    "audio/mp4", "audio/x-m4a", "audio/mp3",
}

app = FastAPI(
    title="Khayt 3D AI API",
    description="AI design assistant for the Khayt 3D embroidery studio.",
    version="1.0.0",
)

origins = [
    item.strip()
    for item in os.getenv("CORS_ORIGINS", "*").split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="مفتاح OPENAI_API_KEY غير مضبوط في إعدادات خادم Render.",
        )
    return OpenAI(api_key=key, timeout=120.0, max_retries=2)

def cloudflare_credentials() -> tuple[str, str]:
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not account_id or not token:
        raise HTTPException(status_code=503, detail="أكمل إعداد Cloudflare AI بإضافة CLOUDFLARE_ACCOUNT_ID وCLOUDFLARE_API_TOKEN في Render.")
    return account_id, token


async def cloudflare_run(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    account_id, token = cloudflare_credentials()
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    try:
        async with httpx.AsyncClient(timeout=150.0) as session:
            response = await session.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload)
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="وصلت إلى الحصة المجانية اليومية لـ Cloudflare AI. انتظر إعادة ضبط الحصة اليومية ثم أعد المحاولة.")
        if response.status_code in (401, 403):
            logger.warning("Cloudflare AI rejected credentials or model access.")
            raise HTTPException(status_code=503, detail="تعذر التحقق من صلاحيات Cloudflare AI. راجع Account ID وصلاحية Workers AI في المفتاح.")
        response.raise_for_status()
        data = response.json()
        if not data.get("success", True):
            raise HTTPException(status_code=502, detail="أعادت خدمة Cloudflare AI خطأ أثناء معالجة الطلب.")
        result = data.get("result")
        return result if isinstance(result, dict) else {"response": result}
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        logger.warning("Cloudflare AI request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="تعذر الاتصال بخدمة Cloudflare AI. حاول مرة أخرى.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="أعادت Cloudflare استجابة غير مفهومة.") from exc


class Point(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    curve: bool = False


class EmbroideryObject(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    type: str
    points: list[Point] = Field(min_length=2, max_length=100)
    closed: bool = False
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    fill: str = "tatami"
    spacing: float = Field(default=0.45, ge=0.2, le=1.2)
    stitch_length: float = Field(default=3.2, ge=0.5, le=7)
    max_stitch_length: float = Field(default=7, ge=1, le=12)
    angle: float = Field(default=0, ge=-180, le=180)
    satin_width: float = Field(default=3, ge=0.5, le=12)


class DSTStitch(BaseModel):
    x: float = Field(default=0, ge=0, le=1200)
    y: float = Field(default=0, ge=0, le=1200)
    jump: bool = False
    command: str | None = None
    color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")


class DSTExportRequest(BaseModel):
    name: str = Field(default="Khayt Design", min_length=1, max_length=80)
    width_mm: float = Field(gt=0, le=1200)
    height_mm: float = Field(gt=0, le=1200)
    initial_color: str = Field(default="#176447", pattern=r"^#[0-9A-Fa-f]{6}$")
    stitches: list[DSTStitch] = Field(min_length=2, max_length=150000)


ALLOWED_TYPES = {"tatami", "satin", "run", "triple", "backstitch"}




def parse_model_json(raw: str) -> dict[str, Any] | None:
    """Parse model JSON, including JSON wrapped in common provider response envelopes."""
    if not raw:
        return None

    decoder = json.JSONDecoder()
    pending: list[str] = [raw.strip()]
    seen: set[str] = set()

    while pending:
        text = pending.pop(0).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned = text.replace("```json", "").replace("```", "").strip()

        try:
            value = json.loads(cleaned)
            if isinstance(value, dict):
                # Some inference gateways wrap the model's actual JSON in a string field.
                if isinstance(value.get("objects"), list) and value["objects"]:
                    return value
                for key in ("response", "text", "output", "content", "generated_text"):
                    nested = value.get(key)
                    if isinstance(nested, str):
                        pending.append(nested)
                    elif isinstance(nested, dict):
                        pending.append(json.dumps(nested, ensure_ascii=False))
                if isinstance(value.get("choices"), list):
                    pending.append(json.dumps(value["choices"], ensure_ascii=False))
                # Keep a structurally valid object as a last resort.
                if not pending:
                    return value
            elif isinstance(value, str):
                pending.append(value)
        except (json.JSONDecodeError, TypeError):
            pass

        # Scan for embedded JSON objects, including responses prefixed by model tokens.
        for index, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                value, _end = decoder.raw_decode(cleaned[index:])
                if isinstance(value, dict):
                    if isinstance(value.get("objects"), list) and value["objects"]:
                        return value
                    for key in ("response", "text", "output", "content", "generated_text"):
                        nested = value.get(key)
                        if isinstance(nested, str):
                            pending.append(nested)
                break
            except json.JSONDecodeError:
                continue

    return None

def extract_model_text(result: dict[str, Any]) -> str:
    """Extract generated text from Cloudflare's response, including nested gateway envelopes."""
    preferred = ("response", "text", "output", "generated_text", "content", "choices", "message")

    def walk(value: Any, depth: int = 0) -> str:
        if depth > 8 or value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            for item in value:
                found = walk(item, depth + 1)
                if found:
                    return found
            return ""
        if isinstance(value, dict):
            if isinstance(value.get("objects"), list) and value["objects"]:
                return json.dumps(value, ensure_ascii=False)
            for key in preferred:
                if key in value:
                    found = walk(value[key], depth + 1)
                    if found:
                        return found
            # Some providers return a nested result object under an undocumented key.
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    found = walk(nested, depth + 1)
                    if found:
                        return found
        return ""

    return walk(result)


def polygon_has_self_intersection(points: list[dict[str, Any]]) -> bool:
    def orient(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any]) -> float:
        return (b["x"] - a["x"]) * (c["y"] - a["y"]) - (b["y"] - a["y"]) * (c["x"] - a["x"])

    count = len(points)
    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        for j in range(i + 1, count):
            if j == i or j == (i + 1) % count or (j + 1) % count == i:
                continue
            c, d = points[j], points[(j + 1) % count]
            o1, o2 = orient(a, b, c), orient(a, b, d)
            o3, o4 = orient(c, d, a), orient(c, d, b)
            if o1 * o2 < -1e-10 and o3 * o4 < -1e-10:
                return True
    return False


def normalize_design(payload: dict[str, Any], width: float, height: float) -> dict[str, Any]:
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not 3 <= len(raw_objects) <= 80:
        raise HTTPException(status_code=502, detail="يجب أن يحتوي التصميم على 3 إلى 80 كائناً هندسياً صالحاً.")

    objects: list[dict[str, Any]] = []
    geometry_fingerprints: set[tuple[tuple[float, float], ...]] = set()
    for index, raw in enumerate(raw_objects, start=1):
        if not isinstance(raw, dict):
            raise HTTPException(status_code=502, detail=f"الكائن {index} ليس بنية هندسية صحيحة.")
        kind = str(raw.get("type", ""))
        if kind not in ALLOWED_TYPES:
            raise HTTPException(status_code=502, detail=f"نوع غرزة غير مدعوم في الكائن {index}: {kind}. لم يتم تغيير النوع تلقائياً.")

        points = raw.get("points", [])
        if not isinstance(points, list) or not 4 <= len(points) <= 100:
            raise HTTPException(status_code=502, detail=f"عدد نقاط الكائن {index} يجب أن يكون بين 4 و40.")
        clean_points = []
        for point in points:
            if not isinstance(point, dict):
                raise HTTPException(status_code=502, detail=f"نقطة غير صالحة في الكائن {index}.")
            try:
                x = float(point["x"])
                y = float(point["y"])
            except (KeyError, TypeError, ValueError):
                raise HTTPException(status_code=502, detail=f"إحداثيات غير رقمية في الكائن {index}.")
            if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
                raise HTTPException(status_code=502, detail=f"إحداثيات الكائن {index} خارج النطاق 0 إلى 1.")
            clean_points.append({
                "x": round(x, 5),
                "y": round(y, 5),
                "curve": bool(point.get("curve", False)),
            })

        if not isinstance(raw.get("closed"), bool):
            raise HTTPException(status_code=502, detail=f"حالة الإغلاق مفقودة أو غير منطقية في الكائن {index}.")
        closed = raw["closed"]
        if closed and (kind != "tatami" or len(clean_points) < 9):
            raise HTTPException(status_code=502, detail=f"الكائن {index} يحتاج شكلاً مغلقاً من نوع tatami وبـ9 نقاط على الأقل.")
        if not closed and (kind not in {"satin", "run", "triple", "backstitch"} or len(clean_points) < 4):
            raise HTTPException(status_code=502, detail=f"الكائن {index} يحتاج مساراً مفتوحاً صالحاً من 4 نقاط على الأقل.")
        unique_points = {(p["x"], p["y"]) for p in clean_points}
        if len(unique_points) < (6 if closed else 4):
            raise HTTPException(status_code=502, detail=f"نقاط الكائن {index} مكررة ولا تصنع هندسة كافية.")
        if closed:
            area = abs(sum(clean_points[i]["x"] * clean_points[(i + 1) % len(clean_points)]["y"] - clean_points[(i + 1) % len(clean_points)]["x"] * clean_points[i]["y"] for i in range(len(clean_points))) / 2)
            if area < 0.00005:
                raise HTTPException(status_code=502, detail=f"مساحة الكائن {index} صغيرة جداً أو نقاطه على خط واحد.")
            if polygon_has_self_intersection(clean_points):
                raise HTTPException(status_code=502, detail=f"حدود الكائن {index} تتقاطع مع نفسها؛ لا يمكن تحويلها بأمان إلى غرز.")
        fingerprint = tuple(sorted((round(p["x"], 4), round(p["y"], 4)) for p in clean_points))
        if fingerprint in geometry_fingerprints:
            raise HTTPException(status_code=502, detail=f"الكائن {index} يكرر هندسة كائن آخر دون اختلاف فعلي.")
        geometry_fingerprints.add(fingerprint)

        color = str(raw.get("color", ""))
        if not (len(color) == 7 and color.startswith("#")):
            raise HTTPException(status_code=502, detail=f"لون غير صالح في الكائن {index}.")
        try:
            int(color[1:], 16)
        except ValueError:
            raise HTTPException(status_code=502, detail=f"لون غير صالح في الكائن {index}.")

        def bounded(key: str, default: float, low: float, high: float) -> float:
            try:
                value = float(raw.get(key, default))
            except (TypeError, ValueError):
                value = default
            return round(max(low, min(high, value)), 3)

        objects.append({
            "name": str(raw.get("name") or f"كائن {index}")[:80],
            "type": kind,
            "points": clean_points,
            "closed": closed,
            "color": color,
            "fill": {"tatami": "tatami", "satin": "satin", "run": "run", "triple": "triple", "backstitch": "backstitch"}[kind],
            "spacing": bounded("spacing", 0.45, 0.2, 1.2),
            "stitch_length": bounded("stitch_length", 3.2, 0.5, 7),
            "max_stitch_length": bounded("max_stitch_length", 7, 1, 12),
            "angle": bounded("angle", 0, -180, 180),
            "satin_width": bounded("satin_width", 3, 0.5, 12),
        })

    if not objects:
        raise HTTPException(
            status_code=502,
            detail="المخطط الناتج لا يحتوي على أشكال هندسية صالحة. جرّب وصفاً أكثر تحديداً.",
        )

    summary = str(payload.get("summary") or "مقترح تصميم تطريز قابل للمراجعة والتعديل.")[:1200]
    return {
        "ok": True,
        "summary": summary,
        "objects": objects,
        "fabric": {"width_mm": width, "height_mm": height},
        "model": OPENAI_MODEL,
    }


def _resample_geometry(points: list[dict[str, Any]], count: int, closed: bool) -> list[dict[str, Any]]:
    """Evenly resample a polyline so embroidery geometry has stable point density."""
    clean: list[dict[str, Any]] = []
    for point in points:
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            continue
        if not clean or math.hypot(x - clean[-1]["x"], y - clean[-1]["y"]) > 1e-5:
            clean.append({"x": x, "y": y, "curve": bool(point.get("curve", False))})
    if closed and len(clean) > 2 and math.hypot(clean[0]["x"] - clean[-1]["x"], clean[0]["y"] - clean[-1]["y"]) < 1e-5:
        clean.pop()
    if len(clean) < (3 if closed else 2):
        return []
    segments = []
    total = 0.0
    limit = len(clean) if closed else len(clean) - 1
    for index in range(limit):
        a, b = clean[index], clean[(index + 1) % len(clean)]
        length = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        segments.append((a, b, total, length))
        total += length
    if total < 1e-5:
        return []
    result = []
    for i in range(count):
        distance = total * (i / count if closed else i / max(1, count - 1))
        chosen = segments[-1]
        for segment in segments:
            if distance <= segment[2] + segment[3] or segment is segments[-1]:
                chosen = segment
                break
        a, b, start, length = chosen
        ratio = 0.0 if length <= 1e-9 else max(0.0, min(1.0, (distance - start) / length))
        result.append({
            "x": round(a["x"] + (b["x"] - a["x"]) * ratio, 6),
            "y": round(a["y"] + (b["y"] - a["y"]) * ratio, 6),
            "curve": bool(a.get("curve", False) or b.get("curve", False)),
        })
    return result


def repair_geometry_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Repair common LLM geometry formatting defects without inventing new objects."""
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
        return payload
    repaired: list[dict[str, Any]] = []
    fingerprints: set[tuple[tuple[float, float], ...]] = set()
    for raw in payload["objects"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("points"), list):
            continue
        closed_value = raw.get("closed")
        closed = closed_value if isinstance(closed_value, bool) else raw.get("type") == "tatami"
        points = raw["points"]
        valid_points = []
        for point in points:
            if not isinstance(point, dict):
                continue
            try:
                x, y = float(point["x"]), float(point["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1:
                valid_points.append({"x": x, "y": y, "curve": bool(point.get("curve", False))})
        if closed:
            # LLMs occasionally return polygon vertices out of perimeter order.
            # Angular ordering converts that point cloud into a simple outline.
            if len(valid_points) >= 3:
                cx = sum(p["x"] for p in valid_points) / len(valid_points)
                cy = sum(p["y"] for p in valid_points) / len(valid_points)
                valid_points.sort(key=lambda p: math.atan2(p["y"] - cy, p["x"] - cx))
            kind = "tatami"
            valid_points = _resample_geometry(valid_points, 16, True)
        else:
            kind = raw.get("type") if raw.get("type") in {"satin", "run", "triple", "backstitch"} else "run"
            valid_points = _resample_geometry(valid_points, max(4, min(10, len(valid_points))), False)
        if not valid_points:
            continue
        fingerprint = tuple(sorted((round(p["x"], 4), round(p["y"], 4)) for p in valid_points))
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        item = dict(raw)
        item.update({
            "type": kind,
            "closed": closed,
            "points": valid_points,
        })
        repaired.append(item)
    result = dict(payload)
    result["objects"] = repaired
    return result


def fallback_flower_design() -> dict[str, Any]:
    """Deterministic, editable flower geometry used only when the AI fails."""
    def ellipse(cx: float, cy: float, rx: float, ry: float, count: int = 20, rotation: float = 0) -> list[dict[str, Any]]:
        points = []
        angle0 = math.radians(rotation)
        for i in range(count):
            t = 2 * math.pi * i / count
            x = rx * math.cos(t)
            y = ry * math.sin(t)
            points.append({
                "x": round(max(0.02, min(0.98, cx + x * math.cos(angle0) - y * math.sin(angle0))), 5),
                "y": round(max(0.02, min(0.98, cy + x * math.sin(angle0) + y * math.cos(angle0))), 5),
                "curve": False,
            })
        return points

    objects: list[dict[str, Any]] = []
    # Five red petals around the flower center.
    for i in range(5):
        angle = -90 + i * 72
        radians = math.radians(angle)
        cx = 0.5 + 0.145 * math.cos(radians)
        cy = 0.34 + 0.145 * math.sin(radians)
        objects.append({
            "name": f"بتلة {i + 1}", "type": "tatami",
            "points": ellipse(cx, cy, 0.083, 0.125, 24, angle + 90),
            "closed": True, "color": "#D92D45", "fill": "tatami",
            "angle": 45,
        })
    objects.extend([
        {
            "name": "قلب الوردة", "type": "tatami",
            "points": ellipse(0.5, 0.34, 0.065, 0.065, 24),
            "closed": True, "color": "#F5C542", "fill": "tatami",
        },
        {
            "name": "ساق خضراء", "type": "run",
            "points": [
                {"x": 0.5, "y": 0.43, "curve": False},
                {"x": 0.505, "y": 0.57, "curve": False},
                {"x": 0.495, "y": 0.72, "curve": False},
                {"x": 0.5, "y": 0.9, "curve": False},
            ],
            "closed": False, "color": "#23834B", "fill": "satin",
            "stitch_length": 2.5,
        },
        {
            "name": "ورقة يسرى", "type": "tatami",
            "points": [
                {"x": 0.495, "y": 0.68, "curve": False},
                {"x": 0.39, "y": 0.61, "curve": False},
                {"x": 0.32, "y": 0.64, "curve": False},
                {"x": 0.37, "y": 0.72, "curve": False},
                {"x": 0.455, "y": 0.73, "curve": False},
            ],
            "closed": True, "color": "#23834B", "fill": "tatami",
        },
        {
            "name": "ورقة يمنى", "type": "tatami",
            "points": [
                {"x": 0.5, "y": 0.78, "curve": False},
                {"x": 0.59, "y": 0.7, "curve": False},
                {"x": 0.68, "y": 0.72, "curve": False},
                {"x": 0.62, "y": 0.8, "curve": False},
                {"x": 0.53, "y": 0.83, "curve": False},
            ],
            "closed": True, "color": "#23834B", "fill": "tatami",
        },
    ])
    return {
        "summary": "تصميم وردة بخمس بتلات وساق وورقتين. استُخدم قالب هندسي احتياطي لأن نموذج الذكاء الاصطناعي لم يُرجع هندسة صالحة؛ يمكنك تعديل الأشكال والألوان داخل البرنامج.",
        "objects": objects,
    }


def standard_tulip_geometry() -> dict[str, Any]:
    """Create a deterministic tulip from smooth sampled Bézier outlines."""
    def bezier(p0, p1, p2, p3, count=10):
        points = []
        for i in range(count):
            t = i / count
            u = 1.0 - t
            x = u**3*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t**3*p3[0]
            y = u**3*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t**3*p3[1]
            points.append({"x": round(x, 6), "y": round(y, 6), "curve": True})
        return points

    def petal(cx, cy, rx, ry, rotation=0):
        segments = [
            ((cx, cy + ry), (cx-rx*.95, cy+ry*.55), (cx-rx*1.05, cy-ry*.65), (cx-rx*.55, cy-ry*.88)),
            ((cx-rx*.55, cy-ry*.88), (cx-rx*.20, cy-ry*1.10), (cx+rx*.20, cy-ry*1.10), (cx+rx*.55, cy-ry*.88)),
            ((cx+rx*.55, cy-ry*.88), (cx+rx*1.05, cy-ry*.65), (cx+rx*.95, cy+ry*.55), (cx, cy+ry)),
        ]
        pts = [p for seg in segments for p in bezier(*seg, count=10)]
        a = math.radians(rotation)
        for p in pts:
            x, y = p["x"]-cx, p["y"]-cy
            p["x"] = round(max(.02, min(.98, cx+x*math.cos(a)-y*math.sin(a))), 6)
            p["y"] = round(max(.02, min(.98, cy+x*math.sin(a)+y*math.cos(a))), 6)
        return pts

    def leaf(cx, cy, rx, ry, rotation):
        segs = [
            ((cx-rx, cy), (cx-rx*.65, cy-ry*.95), (cx+rx*.55, cy-ry*.95), (cx+rx, cy)),
            ((cx+rx, cy), (cx+rx*.55, cy+ry*.95), (cx-rx*.65, cy+ry*.95), (cx-rx, cy)),
        ]
        pts = [p for seg in segs for p in bezier(*seg, count=12)]
        a = math.radians(rotation)
        for p in pts:
            x, y = p["x"]-cx, p["y"]-cy
            p["x"] = round(max(.02, min(.98, cx+x*math.cos(a)-y*math.sin(a))), 6)
            p["y"] = round(max(.02, min(.98, cy+x*math.sin(a)+y*math.cos(a))), 6)
        return pts

    objects = [
        {"name": "بتلة التوليب اليسرى", "type": "tatami", "points": petal(.405, .285, .105, .205, -17), "closed": True, "color": "#C92F50", "fill": "tatami", "angle": 35},
        {"name": "بتلة التوليب اليمنى", "type": "tatami", "points": petal(.595, .285, .105, .205, 17), "closed": True, "color": "#D94360", "fill": "tatami", "angle": -35},
        {"name": "بتلة التوليب الوسطى", "type": "tatami", "points": petal(.5, .255, .112, .235, 0), "closed": True, "color": "#E65B70", "fill": "tatami", "angle": 45},
        {"name": "ورقة التوليب اليسرى", "type": "tatami", "points": leaf(.365, .69, .105, .055, -28), "closed": True, "color": "#287A4B", "fill": "tatami", "angle": 25},
        {"name": "ورقة التوليب اليمنى", "type": "tatami", "points": leaf(.635, .76, .12, .052, 22), "closed": True, "color": "#32915A", "fill": "tatami", "angle": -25},
        {"name": "ساق التوليب", "type": "satin", "points": [
            {"x": .5, "y": .45, "curve": True}, {"x": .49, "y": .56, "curve": True},
            {"x": .505, "y": .68, "curve": True}, {"x": .5, "y": .84, "curve": True}
        ], "closed": False, "color": "#287A4B", "fill": "satin", "satin_width": 3.2, "stitch_length": 2.8},
    ]
    return {"summary": "توليب مرسوم ببتلات منحنية وناعمة، وورقتين طبيعيتين وساق ساتان. هذه هندسة جاهزة للمعاينة والتعديل.", "objects": objects}


def offline_geometric_rosette_design() -> dict[str, Any]:
    """Build an editable oriental eight-petal embroidery motif without any AI provider."""
    def ellipse(cx: float, cy: float, rx: float, ry: float, count: int = 20, rotation: float = 0):
        a = math.radians(rotation)
        points = []
        for i in range(count):
            t = 2 * math.pi * i / count
            x, y = rx * math.cos(t), ry * math.sin(t)
            points.append({
                "x": round(max(.02, min(.98, cx + x * math.cos(a) - y * math.sin(a))), 6),
                "y": round(max(.02, min(.98, cy + x * math.sin(a) + y * math.cos(a))), 6),
                "curve": True,
            })
        return points

    objects: list[dict[str, Any]] = []
    colors = ["#B88732", "#9D263B", "#287A4B", "#F1D58A"]
    # Eight individually editable petals, rotated around a shared center.
    for i in range(8):
        angle = -90 + i * 45
        rad = math.radians(angle)
        objects.append({
            "name": f"بتلة هندسية {i + 1}", "type": "tatami",
            "points": ellipse(.5 + .125 * math.cos(rad), .47 + .125 * math.sin(rad), .068, .125, 24, angle - 90),
            "closed": True, "color": colors[1 if i % 2 == 0 else 0],
            "fill": "tatami", "spacing": .42, "stitch_length": 3.0, "angle": 45 if i % 2 == 0 else -45,
        })
    # Eight narrow leaves form a second ornamental ring between the petals and edge.
    for i in range(8):
        angle = -67.5 + i * 45
        rad = math.radians(angle)
        objects.append({
            "name": f"ورقة زخرفية {i + 1}", "type": "tatami",
            "points": ellipse(.5 + .255 * math.cos(rad), .47 + .255 * math.sin(rad), .067, .027, 20, angle),
            "closed": True, "color": colors[2], "fill": "tatami",
            "spacing": .42, "stitch_length": 2.8, "angle": angle,
        })
    objects.append({
        "name": "قلب الزهرة", "type": "tatami",
        "points": ellipse(.5, .47, .064, .064, 24), "closed": True,
        "color": colors[3], "fill": "tatami", "spacing": .38, "stitch_length": 2.5,
    })
    for i, angle in enumerate([0, 90, 180, 270]):
        rad = math.radians(angle)
        objects.append({
            "name": f"نقطة إطار {i + 1}", "type": "tatami",
            "points": ellipse(.5 + .345 * math.cos(rad), .47 + .345 * math.sin(rad), .025, .025, 16),
            "closed": True, "color": colors[0], "fill": "tatami", "spacing": .4,
        })
    return {
        "summary": "نقش زهرة شرقية هندسية متناظرة بثماني بتلات، وثماني أوراق زخرفية، وقلب مركزي ولمسات إطار. أُنشئ محلياً دون اتصال بأي خدمة ذكاء اصطناعي، وجميع أجزائه قابلة للتحرير.",
        "objects": objects,
    }


def offline_design_for_prompt(prompt: str) -> dict[str, Any]:
    """Safe local fallback when a hosted text model is temporarily rate-limited."""
    if re.search(r"(توليب|tulip)", prompt, re.IGNORECASE):
        return standard_tulip_geometry()
    # A deterministic ornamental motif is preferable to blocking all text design work.
    result = offline_geometric_rosette_design()
    result["summary"] = (
        "تعذر الوصول إلى نموذج الذكاء الاصطناعي بسبب حد الاستخدام؛ لذلك أنشأ البرنامج "
        "نقشاً زخرفياً هندسياً محلياً قابلاً للتحرير كبديل مؤقت. قد يحتاج إلى تعديل ليتطابق "
        "تماماً مع وصفك الأصلي."
    )
    return result


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "ok": True,
        "service": APP_NAME,
        "message": "Khayt 3D AI API is running.",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": APP_NAME,
        "api_key_configured": bool(os.getenv("CLOUDFLARE_API_TOKEN", "").strip()) if AI_PROVIDER == "cloudflare" else bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "provider": AI_PROVIDER,
        "model": CF_TEXT_MODEL if AI_PROVIDER == "cloudflare" else OPENAI_MODEL,
    }




def trace_line_art_image(raw_image: bytes, width_mm: float, height_mm: float) -> dict[str, Any]:
    """Trace monochrome artwork into editable, unfilled running-stitch paths without AI."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        logger.exception("OpenCV is required for deterministic line-art tracing")
        raise HTTPException(status_code=503, detail="ميزة تتبّع الخطوط تحتاج مكتبة معالجة الصور؛ أعد نشر الخادم بعد تحديث المتطلبات.") from exc

    image = cv2.imdecode(np.frombuffer(raw_image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=415, detail="تعذر قراءة الصورة. استخدم PNG أو JPG أو WEBP.")
    h, w = image.shape[:2]
    if w < 20 or h < 20:
        raise HTTPException(status_code=422, detail="أبعاد الصورة صغيرة جداً لتتبّع الخطوط.")

    if max(h, w) > 1800:
        factor = 1800 / max(h, w)
        image = cv2.resize(image, (max(20, int(w * factor)), max(20, int(h * factor))), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold retains thin black strokes on white or lightly textured paper.
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 3:
            clean[labels == label] = 255

    # Bridge only tiny one-pixel breaks in scanned/photographed line art before
    # skeletonization. A small elliptical kernel avoids broadly merging nearby motifs.
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)

    ys, xs = np.where(clean > 0)
    if len(xs) < 30:
        raise HTTPException(status_code=422, detail="لم أعثر على خطوط داكنة كافية في الصورة.")

    # Crop to the drawing and keep its original aspect ratio on the embroidery canvas.
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    drawing = clean[y0:y1, x0:x1]
    dh, dw = drawing.shape

    # Morphological skeletonization reduces each outlined pen stroke to a single centerline.
    work = drawing.copy()
    skeleton = np.zeros_like(work)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max(dw, dh)):
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break

    yy, xx = np.where(skeleton > 0)
    pixels = set(zip(xx.tolist(), yy.tolist()))
    if len(pixels) < 20:
        raise HTTPException(status_code=422, detail="تعذر استخراج خطوط متصلة من الصورة.")

    offsets = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    adjacency: dict[tuple[int,int], list[tuple[int,int]]] = {}
    for p in pixels:
        x, y = p
        ns = []
        for dx, dy in offsets:
            q = (x + dx, y + dy)
            if q not in pixels:
                continue
            # Avoid counting a diagonal shortcut when an orthogonal connection exists.
            if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                continue
            ns.append(q)
        adjacency[p] = ns

    visited: set[tuple[tuple[int,int], tuple[int,int]]] = set()
    paths: list[list[tuple[int,int]]] = []

    def edge(a: tuple[int,int], b: tuple[int,int]) -> tuple[tuple[int,int], tuple[int,int]]:
        return (a, b) if a <= b else (b, a)

    # Start at endpoints and junctions, splitting paths at branch points.
    for start, neighbors in adjacency.items():
        if len(neighbors) == 2:
            continue
        for nxt in neighbors:
            first_edge = edge(start, nxt)
            if first_edge in visited:
                continue
            visited.add(first_edge)
            path = [start, nxt]
            prev, cur = start, nxt
            while len(adjacency[cur]) == 2:
                candidates = [q for q in adjacency[cur] if q != prev and edge(cur, q) not in visited]
                if not candidates:
                    break
                q = candidates[0]
                visited.add(edge(cur, q))
                path.append(q)
                prev, cur = cur, q
            if len(path) >= 8:
                paths.append(path)

    # Also collect closed loops, which have no endpoints or junctions.
    for start, neighbors in adjacency.items():
        for nxt in neighbors:
            if edge(start, nxt) in visited:
                continue
            visited.add(edge(start, nxt))
            path = [start, nxt]
            prev, cur = start, nxt
            while True:
                candidates = [q for q in adjacency[cur] if q != prev and edge(cur, q) not in visited]
                if not candidates:
                    break
                q = candidates[0]
                visited.add(edge(cur, q))
                path.append(q)
                prev, cur = cur, q
                if cur == start:
                    break
            if len(path) >= 8:
                paths.append(path)

    objects: list[dict[str, Any]] = []
    pad = 0.025
    scale = max(dw, dh)
    for index, path in enumerate(paths, start=1):
        pts = np.asarray(path, dtype=np.float32).reshape((-1, 1, 2))
        perimeter = cv2.arcLength(pts, False)
        if perimeter < 18:
            continue
        simplified = cv2.approxPolyDP(pts, max(0.8, scale * 0.0015), False).reshape((-1, 2))
        if len(simplified) < 4:
            # Keep enough centerline points for a smooth stitch path.
            indices = np.linspace(0, len(path) - 1, min(8, len(path))).astype(int)
            simplified = np.asarray([path[i] for i in indices], dtype=np.float32)
        if len(simplified) > 96:
            indices = np.linspace(0, len(simplified) - 1, 96).astype(int)
            simplified = simplified[indices]
        points = []
        for px, py in simplified:
            nx = 0.5 + ((float(px) - (dw - 1) / 2) / scale) * (1 - 2 * pad)
            ny = 0.5 + ((float(py) - (dh - 1) / 2) / scale) * (1 - 2 * pad)
            points.append({"x": round(nx, 5), "y": round(ny, 5), "curve": True})
        if len(points) >= 4:
            objects.append({
                "name": f"خط مرسوم {len(objects) + 1}",
                "type": "run",
                "points": points,
                "closed": False,
                "color": "#000000",
                "fill": "run",
                "spacing": 0.45,
                "stitch_length": 2.5,
                "max_stitch_length": 5,
                "angle": 0,
            })

    if len(objects) < 3:
        raise HTTPException(status_code=422, detail="الصورة لم تنتج ثلاثة مسارات خطية واضحة على الأقل. استخدم صورة أوضح بخلفية بيضاء وخطوط داكنة.")
    if len(objects) > 80:
        objects = sorted(objects, key=lambda item: len(item["points"]), reverse=True)[:80]

    traced = normalize_design(
        {"summary": "تم تتبّع خطوط الصورة آلياً دون استخدام Cloudflare AI. المسارات غرزة جارية سوداء بلا تعبئة؛ راجع المسارات قبل الإنتاج.", "objects": objects},
        width_mm, height_mm
    )
    traced["model"] = "OpenCV deterministic line tracing"
    traced["provider"] = "Khayt built-in image tracing"
    traced["generation_mode"] = "image_trace"
    traced["trace_info"] = {
        "source_width_px": w,
        "source_height_px": h,
        "paths": len(traced["objects"]),
        "stitch_type": "run",
        "filled_areas": False,
    }
    return traced


@app.post("/api/ai/design")
async def create_design(
    prompt: str = Form(default=""),
    width_mm: float = Form(default=300, gt=0, le=1200),
    height_mm: float = Form(default=300, gt=0, le=1200),
    fabric: str = Form(default="قماش تطريز"),
    image: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    prompt = prompt.strip()
    if not prompt and image is None:
        raise HTTPException(status_code=400, detail="اكتب وصف التصميم أو أرفق صورة مرجعية.")

    # A tulip has a deterministic, editable Bézier template. Do not spend provider quota
    # on an LLM response that would be replaced by this exact geometry anyway.
    if image is None and re.search(r"(توليب|tulip)", prompt, re.IGNORECASE):
        normalized = normalize_design(standard_tulip_geometry(), width_mm, height_mm)
        normalized["model"] = "قالب توليب هندسي منحني"
        normalized["provider"] = "Khayt built-in geometry"
        normalized["generation_mode"] = "template"
        return normalized

    # Common ornamental requests are generated locally: they never consume daily AI quota.
    if image is None and re.search(r"(زهرة هندسية|نقش هندسي|زخرف|شرقي|ثماني البتلات|ثمانية بتلات|mandala|rosette|medallion|geometric flower)", prompt, re.IGNORECASE):
        normalized = normalize_design(offline_geometric_rosette_design(), width_mm, height_mm)
        normalized["model"] = "Khayt offline geometric generator"
        normalized["provider"] = "Khayt built-in geometry"
        normalized["generation_mode"] = "offline_template"
        return normalized

    client = None if AI_PROVIDER == "cloudflare" else get_client()
    instructions = f"""
أنت مهندس تصميم تطريز آلي. حوّل طلب المستخدم إلى مخطط هندسي ثنائي الأبعاد قابل للتحرير والتحويل إلى غرز.
أبعاد القماش: {width_mm} × {height_mm} مم. نوع القماش: {fabric}.
أخرج JSON فقط بالمفتاحين summary و objects.
objects مصفوفة من 3 إلى 18 كائناً. كل كائن يحتوي name و type و points و closed و color.
للأشكال المغلقة استخدم tatami فقط وبـ9 إلى 24 نقطة. للمسارات المفتوحة استخدم satin أو run أو triple أو backstitch وبـ4 إلى 10 نقاط. لا تستخدم motif أو complex.
كل نقطة كائن يحوي x و y بين 0 و1 و curve قيمة true أو false.
للشكل المغلق: استخدم tatami فقط و9 إلى 24 نقطة على المحيط. لا تستخدم مثلثات أو خماسيات بدائية. اجعل البتلات مستديرة والأوراق طويلة ومدببة بانحناءات طبيعية.
للمسار المفتوح: استخدم 4 إلى 10 نقاط على محور المسار. استخدم satin لمسار عمود ساتان مفتوح بخط مركزي، وrun أو triple أو backstitch للخطوط الرفيعة.
اجعل النقاط موزعة على الشكل وليست متقاربة في موضع واحد. لا تكرر الإحداثيات أو الأشكال.
قسّم التصميم إلى أجزاء منفصلة حسب اللون. استخدم tatami للمساحات العريضة المغلقة، satin لمسارات الأعمدة المفتوحة، وrun أو triple أو backstitch للخطوط الرفيعة.
يمكن إضافة fill و spacing و stitch_length و max_stitch_length و angle و satin_width عند الحاجة.
كل الإحداثيات ضمن 0..1. لا تكتب fusion كنوع. لا تضع Markdown أو شرحاً خارج JSON.
لا تدّعِ أن الناتج ملف DST نهائي؛ إنه مقترح هندسي سيحوّله البرنامج إلى غرز.
"""

    if re.search(r"(توليب|وردة|زهرة|tulip|rose|flower)", prompt, re.IGNORECASE):
        instructions += (
            "\\nقاعدة اكتمال إلزامية لهذا الطلب الزهري: أنشئ ثلاث بتلات منفصلة على الأقل، "
            "وكل بتلة يجب أن تكون كائناً مغلقاً مستقلاً من نوع tatami واسمه يتضمن كلمة بتلة؛ "
            "لا تحتسب حافة satin على أنها بتلة. أنشئ ورقتين خضراوين على الأقل، "
            "وكل ورقة كائن مغلق مستقل tatami واسمه يتضمن ورقة، وأنشئ ساقاً واحدة على الأقل "
            "كائن satin مفتوح واسمه ساق. لا تحذف أو ترقم أجزاء التصميم مع ترك جزء مفقود."
        )


    user_content: list[dict[str, Any]] = [{
        "type": "text",
        "text": prompt or "حلّل الصورة المرجعية وأنشئ منها تصميماً مناسباً للتطريز.",
    }]
    image_data_url: str | None = None
    if image is not None:
        content_type = (image.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="صيغة الصورة يجب أن تكون PNG أو JPG أو WEBP.")
        raw_image = await image.read()
        if not raw_image or len(raw_image) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="حجم الصورة فارغ أو يتجاوز 12 ميجابايت.")
        # Explicit line-art requests use local deterministic tracing and never call a paid/quota-limited AI model.
        if re.search(r"(تتبّع|تتبع|الخطوط|خطوط فقط|خطية|زخرفة خطية|بدون تعبئة|لا تستخدم تعبئة|مساحات بيضاء|running stitch|line art)", prompt, re.IGNORECASE):
            return trace_line_art_image(raw_image, width_mm, height_mm)
        encoded = base64.b64encode(raw_image).decode("ascii")
        image_data_url = f"data:{content_type};base64,{encoded}"
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{content_type};base64,{encoded}", "detail": "high"},
        })

    geometry_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "objects": {
                "type": "array",
                "minItems": 3,
                "maxItems": 18,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ["tatami", "satin", "run", "triple", "backstitch"]},
                        "points": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 40,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x": {"type": "number", "minimum": 0, "maximum": 1},
                                    "y": {"type": "number", "minimum": 0, "maximum": 1},
                                    "curve": {"type": "boolean"}
                                },
                                "required": ["x", "y", "curve"]
                            }
                        },
                        "closed": {"type": "boolean"},
                        "color": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                        "fill": {"type": "string"},
                        "spacing": {"type": "number"},
                        "stitch_length": {"type": "number"},
                        "max_stitch_length": {"type": "number"},
                        "angle": {"type": "number"},
                        "satin_width": {"type": "number"}
                    },
                    "required": ["name", "type", "points", "closed", "color"]
                }
            }
        },
        "required": ["summary", "objects"]
    }

    try:
        if AI_PROVIDER == "cloudflare":
            generation_prompt = prompt or "أنشئ تصميماً زخرفياً مناسباً للتطريز."
            if image_data_url:
                try:
                    vision_result = await cloudflare_run(CF_VISION_MODEL, {
                    "messages": [
                        {"role": "system", "content": "أنت محلل صور تطريز. صف العناصر المرئية والألوان والأجزاء والعلاقات المكانية والحدود بوضوح. لا تخترع عناصر غير موجودة ولا تُخرج إحداثيات."},
                        {"role": "user", "content": [
                            {"type": "text", "text": generation_prompt + "\\nحلل الصورة المرجعية المرفقة لاستخدامها في بناء مخطط تطريز."},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ]},
                    ],
                    "max_tokens": 1200,
                    "temperature": 0.1,
                    })
                except HTTPException as exc:
                    if exc.status_code in {403, 503}:
                        raise HTTPException(status_code=503, detail="تعذر استخدام نموذج تحليل الصور في Cloudflare. تحقق من صلاحية Workers AI، ومن تفعيل نموذج Llama Vision وقبول ترخيص Meta له لأول مرة من حساب Cloudflare.") from exc
                    raise
                image_description = extract_model_text(vision_result)
                if not image_description:
                    raise HTTPException(status_code=502, detail="لم يتمكن نموذج الرؤية من تحليل الصورة المرجعية.")
                generation_prompt += "\\nوصف الصورة المرجعية بعد تحليلها: " + image_description

            try:
                result = await cloudflare_run(CF_TEXT_MODEL, {
                    "messages": [
                        {"role": "system", "content": instructions + "\nأعد JSON صالحاً فقط دون Markdown أو شرح خارج JSON."},
                        {"role": "user", "content": generation_prompt},
                    ],
                    "response_format": {"type": "json_schema", "json_schema": geometry_schema},
                    "max_tokens": 6000,
                    "temperature": 0.15,
                })
            except HTTPException as provider_error:
                if provider_error.status_code == 429 and image is None:
                    logger.warning("Cloudflare AI quota exhausted; using local editable geometry fallback.")
                    normalized = normalize_design(offline_design_for_prompt(prompt), width_mm, height_mm)
                    normalized["model"] = "Khayt offline fallback"
                    normalized["provider"] = "Khayt built-in geometry"
                    normalized["generation_mode"] = "offline_fallback"
                    return normalized
                raise
            raw = extract_model_text(result)
            payload = repair_geometry_payload(parse_model_json(raw))

            def geometry_problem(value: dict[str, Any] | None) -> str | None:
                if not isinstance(value, dict):
                    return "لم يرجع النموذج كائن JSON."
                candidates = value.get("objects")
                if not isinstance(candidates, list) or not 3 <= len(candidates) <= 18:
                    return "عدد العناصر يجب أن يكون من 3 إلى 18."
                seen_geometry: set[tuple[tuple[float, float], ...]] = set()
                for index, item in enumerate(candidates, start=1):
                    if not isinstance(item, dict) or not isinstance(item.get("points"), list):
                        return f"العنصر {index}: بيانات العنصر أو قائمة النقاط مفقودة."
                    points = item["points"]
                    closed = item.get("closed")
                    if not isinstance(closed, bool):
                        return f"العنصر {index}: closed يجب أن تكون true أو false."
                    kind = item.get("type")
                    if closed and (len(points) < 9 or kind != "tatami"):
                        return f"العنصر {index}: الشكل المغلق يحتاج tatami و9 نقاط على الأقل (وصل {len(points)} ونوعه {kind})."
                    if not closed and (len(points) < 4 or kind not in {"satin", "run", "triple", "backstitch"}):
                        return f"العنصر {index}: المسار المفتوح يحتاج 4 نقاط على الأقل ونوع غرزة خطي (وصل {len(points)} ونوعه {kind})."
                    if len(points) > 40:
                        return f"العنصر {index}: تجاوز 40 نقطة."
                    if not all(isinstance(p, dict) and isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float)) for p in points):
                        return f"العنصر {index}: الإحداثيات ليست أرقاماً."
                    if not all(math.isfinite(float(p["x"])) and math.isfinite(float(p["y"])) and 0 <= float(p["x"]) <= 1 and 0 <= float(p["y"]) <= 1 for p in points):
                        return f"العنصر {index}: توجد إحداثيات خارج النطاق 0..1."
                    unique = {(round(float(p["x"]), 4), round(float(p["y"]), 4)) for p in points}
                    if len(unique) < (6 if closed else 4):
                        return f"العنصر {index}: نقاط مكررة ولا تكوّن شكلاً."
                    fingerprint = tuple(sorted(unique))
                    if fingerprint in seen_geometry:
                        return f"العنصر {index}: يكرر هندسة عنصر سابق."
                    seen_geometry.add(fingerprint)
                    if closed:
                        area = abs(sum(float(points[i]["x"]) * float(points[(i + 1) % len(points)]["y"]) - float(points[(i + 1) % len(points)]["x"]) * float(points[i]["y"]) for i in range(len(points))) / 2)
                        if area < 0.00005:
                            return f"العنصر {index}: مساحة الشكل صغيرة جداً."
                        if polygon_has_self_intersection(points):
                            return f"العنصر {index}: حدود الشكل تتقاطع مع نفسها."
                    color = item.get("color")
                    if not isinstance(color, str) or len(color) != 7 or not color.startswith("#"):
                        return f"العنصر {index}: رمز اللون غير صالح."
                    try:
                        int(color[1:], 16)
                    except ValueError:
                        return f"العنصر {index}: رمز اللون غير صالح."
                return None

            def contains_objects(value: dict[str, Any] | None) -> bool:
                if geometry_problem(value) is not None:
                    return False
                if not re.search(r"(توليب|وردة|زهرة|tulip|rose|flower)", prompt, re.IGNORECASE):
                    return True
                objects = value.get("objects", []) if isinstance(value, dict) else []
                filled_petals = [item for item in objects if item.get("closed") is True and item.get("type") == "tatami" and re.search(r"(بتل|بتلة|petal)", str(item.get("name", "")), re.IGNORECASE)]
                filled_leaves = [item for item in objects if item.get("closed") is True and item.get("type") == "tatami" and re.search(r"(ورق|ورقة|leaf)", str(item.get("name", "")), re.IGNORECASE)]
                stems = [item for item in objects if item.get("closed") is False and item.get("type") == "satin" and re.search(r"(ساق|stem)", str(item.get("name", "")), re.IGNORECASE)]
                return len(filled_petals) >= 3 and len(filled_leaves) >= 2 and bool(stems)

            # Retry once with a shorter, internally consistent geometry contract.
            if not contains_objects(payload):
                repair = await cloudflare_run(CF_TEXT_MODEL, {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "أنت مهندس تطريز. أخرج JSON فقط بالمفتاحين summary و objects. "
                                "أنشئ 3 إلى 12 كائناً. كل كائن يحتوي name و type و points و closed و color. "
                                "للأشكال المغلقة tatami فقط؛ للمسارات المفتوحة satin أو run أو triple أو backstitch فقط. لا تستخدم motif أو complex. "
                                "كل نقطة تحوي x و y بين 0 و1 و curve منطقية. "
                                "كل شكل مغلق tatami يحتاج 9 نقاط على الأقل، وكل مسار مفتوح satin أو run أو triple أو backstitch يحتاج 4 نقاط على الأقل. "
                                "لا تستخدم أمثلة مختصرة أو نقاطاً أقل من الحد المطلوب. "
                                "إذا كان هناك خلل محدد في المحاولة السابقة فأصلحه دون تغيير فكرة التصميم."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "أنشئ JSON صالحاً الآن. ارسم الأشكال المطلوبة فعلياً بنقاط كافية وحدود طبيعية. "
                                "اجعل الأشكال المغلقة من 9 إلى 24 نقطة والمسارات المفتوحة من 4 إلى 10 نقاط. "
                                "طلب المستخدم: " + generation_prompt
                                + "\\nسبب رفض المحاولة الأولى: " + str(geometry_problem(payload) or "تصميم الزهرة ناقص: يلزم ثلاث بتلات TATAMI وورقتان TATAMI وساق SATIN.")
                                + "\\nصحح JSON السابق التالي مع الاحتفاظ بفكرته: " + raw[:12000]
                                + "\\nالأبعاد: " + str(width_mm) + " × " + str(height_mm) + " مم. القماش: " + fabric
                            ),
                        },
                    ],
                    "response_format": {"type": "json_schema", "json_schema": geometry_schema},
                    "max_tokens": 4500,
                    "temperature": 0.1,
                })
                repaired_raw = extract_model_text(repair)
                repaired_payload = repair_geometry_payload(parse_model_json(repaired_raw))
                if contains_objects(repaired_payload):
                    payload = repaired_payload
                else:
                    logger.warning(
                        "Cloudflare geometry rejected after retry (initial_length=%s, retry_length=%s, keys=%s).",
                        len(raw), len(repaired_raw),
                        list(repaired_payload.keys()) if isinstance(repaired_payload, dict) else [],
                    )
                    logger.warning("Cloudflare geometry validation reason: %s", geometry_problem(repaired_payload) or "unknown")
                    raise HTTPException(
                        status_code=502,
                        detail="لم يتمكن النموذج من إنشاء هندسة صالحة بعد محاولتين. جرّب وصفاً أبسط.",
                    )

            if re.search(r"(توليب|tulip)", prompt, re.IGNORECASE):
                # Use a controlled smooth tulip silhouette instead of accepting angular LLM polygons.
                payload = standard_tulip_geometry()

            normalized = normalize_design(payload, width_mm, height_mm)
            normalized["model"] = CF_TEXT_MODEL
            normalized["provider"] = "Cloudflare Workers AI"
            normalized["generation_mode"] = "ai"
            return normalized

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.35,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        return normalize_design(payload, width_mm, height_mm)
    except HTTPException:
        raise
    except AuthenticationError as exc:
        logger.warning("OpenAI rejected the configured API key (authentication failed).")
        raise HTTPException(
            status_code=503,
            detail="مفتاح OpenAI المحفوظ في Render غير صالح أو تم إلغاؤه. أنشئ مفتاحاً جديداً واحفظه في OPENAI_API_KEY ثم أعد النشر.",
        ) from exc
    except RateLimitError as exc:
        logger.warning("OpenAI rate limit or quota error during design generation.")
        raise HTTPException(
            status_code=429,
            detail="رفضت OpenAI الطلب بسبب حد الاستخدام أو عدم توفر الرصيد. تحقق من Usage وBilling في حساب OpenAI.",
        ) from exc
    except json.JSONDecodeError as exc:
        logger.exception("AI returned invalid JSON")
        raise HTTPException(status_code=502, detail="أعاد نموذج الذكاء الاصطناعي نتيجة غير مفهومة. أعد المحاولة.") from exc
    except Exception as exc:
        logger.exception("Design generation failed")
        raise HTTPException(
            status_code=502,
            detail="تعذر إنشاء التصميم عبر خدمة الذكاء الاصطناعي. تحقق من رصيد المفتاح وصلاحيته ثم راجع سجلات Render.",
        ) from exc


@app.post("/api/export/{format_name}")
async def export_embroidery(request: DSTExportRequest, format_name: str) -> Response:
    """Encode validated stitch coordinates in a supported machine embroidery format."""
    format_name = format_name.strip().lower()
    writer_names = {"dst": "write_dst", "pes": "write_pes", "jef": "write_jef", "exp": "write_exp", "vp3": "write_vp3"}
    if format_name not in writer_names:
        raise HTTPException(status_code=422, detail="صيغة الملف غير مدعومة حالياً. الصيغ المتاحة: DST وPES وJEF وEXP وVP3.")
    if len(request.stitches) > 120000:
        raise HTTPException(status_code=413, detail="عدد الغرز يتجاوز الحد الآمن للتصدير (120 ألف غرزة).")

    writer = getattr(pyembroidery, writer_names[format_name], None)
    if not callable(writer):
        raise HTTPException(status_code=501, detail=f"محرك الترميز المثبت لا يوفر صيغة {format_name.upper()} حالياً.")

    pattern = pyembroidery.EmbPattern()
    pattern["name"] = request.name[:16]
    pattern += request.initial_color
    stitch_count = 0
    color_count = 0

    try:
        for item in request.stitches:
            if item.command == "color":
                if not item.color:
                    raise HTTPException(status_code=422, detail="تغيير اللون يحتاج رمز لون صالحاً.")
                pattern += item.color
                color_count += 1
                continue
            if item.x > request.width_mm or item.y > request.height_mm:
                raise HTTPException(status_code=422, detail="توجد غرزة خارج حدود مساحة القماش.")
            x = round(item.x * 10, 3)
            y = round(-item.y * 10, 3)
            if item.jump:
                pattern.add_stitch_absolute(pyembroidery.JUMP, x, y)
            else:
                pattern.add_stitch_absolute(pyembroidery.STITCH, x, y)
                stitch_count += 1
        if stitch_count < 2:
            raise HTTPException(status_code=422, detail="لا توجد غرز فعلية كافية لإنشاء ملف تطريز.")
        pattern.add_command(pyembroidery.END)
        output = io.BytesIO()
        if format_name == "dst":
            writer(pattern, output, {"explicit_trim": True, "max_stitch": 121, "max_jump": 121})
        else:
            writer(pattern, output)
        content = output.getvalue()
        if not content or (format_name == "dst" and (len(content) < 515 or len(content) % 3 != 2)):
            logger.error("%s encoder returned an unexpected file length: %s", format_name.upper(), len(content))
            raise HTTPException(status_code=500, detail="تعذر التحقق من بنية ملف التطريز الناتج.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("%s export failed", format_name.upper())
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء إنشاء ملف {format_name.upper()}. راجع سجل الخادم.") from exc

    safe_name = "".join(ch for ch in request.name if ch.isalnum() or ch in " _-").strip()[:48] or "khayt-design"
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.{format_name}"',
            "X-Khayt-Stitch-Count": str(stitch_count),
            "X-Khayt-Color-Changes": str(color_count),
            "X-Khayt-Format": format_name.upper(),
        },
    )


@app.post("/api/ai/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_AUDIO_TYPES and not content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="الملف المرفوع ليس ملفاً صوتياً مدعوماً.")
    audio = await file.read()
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="حجم التسجيل فارغ أو يتجاوز 25 ميجابايت.")

    if AI_PROVIDER == "cloudflare":
        audio_b64 = base64.b64encode(audio).decode("ascii")
        result = await cloudflare_run(CF_AUDIO_MODEL, {"audio": audio_b64, "task": "transcribe"})
        return {"ok": True, "text": result.get("text", "")}

    client = get_client()
    try:
        result = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=(file.filename or "voice-request.webm", audio, content_type or "audio/webm"),
        )
        return {"ok": True, "text": getattr(result, "text", "")}
    except Exception as exc:
        logger.exception("Audio transcription failed")
        raise HTTPException(
            status_code=502,
            detail="تعذر تحويل التسجيل الصوتي إلى نص. تحقق من إعدادات خدمة الصوت ومفتاح API.",
        ) from exc
