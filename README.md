# خيط 3D — استوديو تصميم التطريز بالذكاء الاصطناعي

هذا المستودع يحتوي واجهة محرر التطريز وخادم AI المساعد.

## بنية المشروع
- `index.html`: واجهة محرر خيط 3D.
- `khayt-ai-server/`: خدمة FastAPI لتوليد مقترحات التصميم وتحويل الصوت إلى نص.
- `khayt-ai-server/render.yaml`: إعداد نشر خادم الذكاء الاصطناعي على Render.

## نشر خادم AI على Render
أنشئ Web Service من هذا المستودع، واجعل Root Directory هو `khayt-ai-server`:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health Check: `/health`

أضف متغيرات البيئة:
- `OPENAI_API_KEY` — مفتاح API الخاص بك (في Render فقط، لا ترفعه إلى GitHub).
- `OPENAI_MODEL=gpt-4.1`
- `OPENAI_TRANSCRIBE_MODEL=gpt-4o-mini-transcribe`
- `CORS_ORIGINS` — نطاق الواجهة المنشورة، إذا كانت منفصلة عن الخادم.

## تشغيل محلي
راجع `khayt-ai-server/README-AR.md` للتعليمات العربية التفصيلية.

## تنبيه
مخرجات AI أشكال هندسية أولية وليست رقمنة تطريز صناعية معتمدة. افحص التصميم وجرّب عينة قبل الإنتاج.