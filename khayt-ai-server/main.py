from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("khayt-ai")

APP_NAME = "khayt-ai"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
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


ALLOWED_TYPES = {
    "tatami", "motif", "satin", "run", "triple", "backstitch",
    "widezig", "applique", "ring", "star", "circle", "inputA", "inputB",
}


def normalize_design(payload: dict[str, Any], width: float, height: float) -> dict[str, Any]:
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise HTTPException(status_code=502, detail="لم ينتج الذكاء الاصطناعي أي كائنات تصميم.")
    if len(raw_objects) > 40:
        raw_objects = raw_objects[:40]

    objects: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_objects, start=1):
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("type", "tatami"))
        if kind not in ALLOWED_TYPES:
            kind = "tatami"

        points = raw.get("points", [])
        if not isinstance(points, list):
            continue
        clean_points = []
        for point in points[:100]:
            if not isinstance(point, dict):
                continue
            try:
                x = max(0.0, min(1.0, float(point.get("x", 0.5))))
                y = max(0.0, min(1.0, float(point.get("y", 0.5))))
            except (TypeError, ValueError):
                continue
            clean_points.append({
                "x": round(x, 5),
                "y": round(y, 5),
                "curve": bool(point.get("curve", False)),
            })

        closed = bool(raw.get("closed", False))
        if len(clean_points) < (3 if closed else 2):
            continue

        color = str(raw.get("color", "#176447"))
        if not (len(color) == 7 and color.startswith("#")):
            color = "#176447"
        try:
            int(color[1:], 16)
        except ValueError:
            color = "#176447"

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
            "fill": str(raw.get("fill") or ("satin" if kind == "satin" else "tatami")),
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
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "model": OPENAI_MODEL,
    }


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

    client = get_client()
    instructions = f"""
أنت مهندس تصميم تطريز آلي محترف. حوّل طلب المستخدم إلى مخطط هندسي ثنائي الأبعاد
مكوّن من أشكال ومسارات قابلة للتحويل إلى غرز في برنامج خيط 3D.
أبعاد القماش: {width_mm} × {height_mm} مم. نوع القماش: {fabric}.
أعد JSON فقط بالمفاتيح:
summary: وصف عربي موجز.
objects: مصفوفة من 1 إلى 18 كائناً. لكل كائن:
name (اسم عربي قصير)، type (واحد من tatami,motif,satin,run,triple,backstitch,widezig,applique,ring,star,circle,inputA,inputB)،
points (مصفوفة نقاط بين 0 و1 نسبةً إلى عرض وارتفاع القماش، كل نقطة {{"x":0.5,"y":0.5,"curve":false}})،
closed (boolean)، color (#RRGGBB)، fill، spacing (0.2 إلى 1.2)،
stitch_length (0.5 إلى 7 مم)، max_stitch_length (1 إلى 12 مم)،
angle (-180 إلى 180)، satin_width (0.5 إلى 12 مم).
ارسم حدوداً هندسية معقولة ومبسطة، ولا تضع أي نقطة خارج 0..1.
الأشكال المغلقة تحتاج 3 نقاط على الأقل، والمسارات المفتوحة نقطتين على الأقل.
قسّم التصميم إلى كائنات منفصلة حسب الألوان والأجزاء، ووازن بين التفاصيل وسهولة التطريز.
لا تدّعِ أن المخطط ملف DST نهائي؛ إنه مقترح هندسي سيُراجع ويحوّل إلى غرز داخل البرنامج.
"""

    user_content: list[dict[str, Any]] = [{
        "type": "text",
        "text": prompt or "حلّل الصورة المرجعية وأنشئ منها تصميماً مناسباً للتطريز.",
    }]
    if image is not None:
        content_type = (image.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="صيغة الصورة يجب أن تكون PNG أو JPG أو WEBP.")
        raw_image = await image.read()
        if not raw_image or len(raw_image) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="حجم الصورة فارغ أو يتجاوز 12 ميجابايت.")
        encoded = base64.b64encode(raw_image).decode("ascii")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{content_type};base64,{encoded}", "detail": "high"},
        })

    try:
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
    except json.JSONDecodeError as exc:
        logger.exception("AI returned invalid JSON")
        raise HTTPException(status_code=502, detail="أعاد نموذج الذكاء الاصطناعي نتيجة غير مفهومة. أعد المحاولة.") from exc
    except Exception as exc:
        logger.exception("Design generation failed")
        raise HTTPException(
            status_code=502,
            detail="تعذر إنشاء التصميم عبر خدمة الذكاء الاصطناعي. تحقق من رصيد المفتاح وصلاحيته ثم راجع سجلات Render.",
        ) from exc


@app.post("/api/ai/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_AUDIO_TYPES and not content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="الملف المرفوع ليس ملفاً صوتياً مدعوماً.")
    audio = await file.read()
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="حجم التسجيل فارغ أو يتجاوز 25 ميجابايت.")

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
