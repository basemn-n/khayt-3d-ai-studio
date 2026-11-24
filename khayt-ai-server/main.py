from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
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


ALLOWED_TYPES = {
    "tatami", "motif", "satin", "run", "triple", "backstitch",
    "widezig", "applique", "ring", "star", "circle", "inputA", "inputB",
}




def parse_model_json(raw: str) -> dict[str, Any] | None:
    """Parse a JSON object even when the model wraps it in prose or Markdown."""
    if not raw:
        return None
    text = raw.strip().replace("```json", "").replace("```", "").strip()
    decoder = json.JSONDecoder()
    # First try the complete response, then scan for the first valid JSON object.
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except (json.JSONDecodeError, TypeError):
        pass
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


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

    client = None if AI_PROVIDER == "cloudflare" else get_client()
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
قسّم التصميم إلى كائنات منفصلة حسب الألوان والأجزاء، ووازن بين التفاصيل وسهولة التطريز. استخدم tatami للأشكال المغلقة العريضة. استخدم satin فقط للأعمدة الرفيعة والمسارات المفتوحة ذات محور مركزي واضح (closed=false)، مع 2 إلى 40 نقطة على محور العمود. استخدم run للخطوط الرفيعة جداً. لا ترسم عمود satin كحد مغلق.
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
        if AI_PROVIDER == "cloudflare":
            if image is not None:
                raise HTTPException(status_code=501, detail="في الوضع المجاني الحالي، أنشئ التصميم من الوصف النصي. دعم تحليل الصور المرجعية سيضاف بعد اختبار واجهة الرؤية.")
            result = await cloudflare_run(CF_TEXT_MODEL, {
                "messages": [
                    {"role": "system", "content": instructions + "\nأعد JSON صالحاً فقط دون Markdown أو شرح خارج JSON."},
                    {"role": "user", "content": prompt or "أنشئ تصميماً زخرفياً مناسباً للتطريز."},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 4500,
                "temperature": 0.15,
            })
            raw = str(result.get("response") or result.get("text") or "").strip()
            payload = parse_model_json(raw)

            def contains_objects(value: dict[str, Any] | None) -> bool:
                if not isinstance(value, dict):
                    return False
                candidates = value.get("objects")
                if not isinstance(candidates, list) or not candidates:
                    candidates = value.get("elements") or value.get("shapes") or value.get("paths")
                return isinstance(candidates, list) and any(
                    isinstance(item, dict) and isinstance(item.get("points"), list) and len(item["points"]) >= 2
                    for item in candidates
                )

            # Retry if the model returned prose, malformed JSON, or JSON without geometry.
            if not contains_objects(payload):
                repair = await cloudflare_run(CF_TEXT_MODEL, {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "أنت مهندس تطريز ومهمتك إنشاء مخطط هندسي. أخرج JSON فقط دون Markdown. "
                                "يجب أن يحتوي الجذر على summary نص عربي و objects مصفوفة غير فارغة. "
                                "أنشئ من 3 إلى 7 أشكال فعلية. كل عنصر في objects يجب أن يحتوي: "
                                "name نص، type من tatami أو satin أو run أو motif، "
                                "points مصفوفة من 3 إلى 12 نقطة، وكل نقطة {x رقم بين 0 و1, y رقم بين 0 و1, curve false}، "
                                "closed قيمة true أو false، color بصيغة #RRGGBB. "
                                "لا تستخدم مفاتيح بديلة مثل elements أو shapes. لا تترك objects فارغة."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "أنشئ هندسة التصميم من هذا الطلب: " + prompt
                                + "\nالأبعاد بالملليمتر: " + str(width_mm) + " × " + str(height_mm)
                                + "\nنوع القماش: " + fabric
                                + "\nالرد السابق الذي لم يتضمن أشكالاً صالحة (للاستفادة منه فقط): "
                                + raw[:3500]
                            ),
                        },
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 4500,
                    "temperature": 0.1,
                })
                repaired_raw = str(repair.get("response") or repair.get("text") or "").strip()
                repaired_payload = parse_model_json(repaired_raw)
                if contains_objects(repaired_payload):
                    payload = repaired_payload
                else:
                    logger.warning(
                        "Cloudflare response missing geometry after retry (initial_length=%s, retry_length=%s, keys=%s).",
                        len(raw), len(repaired_raw),
                        list(repaired_payload.keys()) if isinstance(repaired_payload, dict) else [],
                    )
                    if any(word in prompt.lower() for word in ("وردة", "زهرة", "flower", "rose")):
                        payload = fallback_flower_design()
                        normalized = normalize_design(payload, width_mm, height_mm)
                        normalized["model"] = "built-in-flower-fallback"
                        normalized["provider"] = "Khayt geometry fallback"
                        normalized["generation_mode"] = "template_fallback"
                        return normalized
                    raise HTTPException(
                        status_code=502,
                        detail="لم يُنشئ النموذج الأشكال الهندسية المطلوبة. جرّب وصفاً أبسط أو قالباً جاهزاً.",
                    )

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


@app.post("/api/export/dst")
async def export_dst(request: DSTExportRequest) -> Response:
    """Encode validated stitch coordinates as a Tajima DST embroidery file."""
    if len(request.stitches) > 120000:
        raise HTTPException(status_code=413, detail="عدد الغرز يتجاوز الحد الآمن للتصدير (120 ألف غرزة).")

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
            raise HTTPException(status_code=422, detail="لا توجد غرز فعلية كافية لإنشاء ملف DST.")
        pattern.add_command(pyembroidery.END)

        output = io.BytesIO()
        pyembroidery.write_dst(
            pattern,
            output,
            {"explicit_trim": True, "max_stitch": 121, "max_jump": 121},
        )
        content = output.getvalue()
        if len(content) < 515 or len(content) % 3 != 2:
            logger.error("DST encoder returned an unexpected file length: %s", len(content))
            raise HTTPException(status_code=500, detail="تعذر التحقق من بنية ملف DST الناتج.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("DST export failed")
        raise HTTPException(status_code=500, detail="حدث خطأ أثناء ترميز ملف DST. راجع سجل الخادم.") from exc

    safe_name = "".join(ch for ch in request.name if ch.isalnum() or ch in " _-").strip()[:48] or "khayt-design"
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.dst"',
            "X-Khayt-Stitch-Count": str(stitch_count),
            "X-Khayt-Color-Changes": str(color_count),
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
