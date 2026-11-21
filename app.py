"""
app.py - RojakBrew Unmanned Smart Kiosk Backend Server
Powered by Scikit-Learn Machine Learning Models + Unified Coffee Knowledge Base
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, File, UploadFile, Response, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import edge_tts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="RojakBrew NLP Kiosk API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 1. Load ML Artifacts & Knowledge Base Datasets
# -------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "chatcoffee_intent_classifier.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models", "rojak_intent_classifier.joblib")
SLANG_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "shortform-dict.json")
if not os.path.exists(SLANG_PATH):
    SLANG_PATH = os.path.join(BASE_DIR, "shortform-dict.json")

KB_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "unified_coffee_kb.csv")
if not os.path.exists(KB_PATH):
    KB_PATH = os.path.join(BASE_DIR, "unified_coffee_kb.csv")
if not os.path.exists(KB_PATH):
    KB_PATH = os.path.join(BASE_DIR, "data", "unified_coffee_kb.csv")

# Load model
if os.path.exists(MODEL_PATH):
    ml_artifact = joblib.load(MODEL_PATH)
    tfidf_vec = ml_artifact["vectorizer"]
    intent_model = ml_artifact["model"]
    classes = ml_artifact["classes"]
    print(f"[ML Pipeline] Loaded champion intent model from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_and_export_model.py first.")

# Load slang dictionary
slang_dict = {}
if os.path.exists(SLANG_PATH):
    with open(SLANG_PATH, "r", encoding="utf-8") as f:
        raw_slang = json.load(f)
        if isinstance(raw_slang, dict):
            for k, v in raw_slang.items():
                if isinstance(v, dict) and k != 'intensifiers':
                    slang_dict.update(v)
                elif isinstance(v, str):
                    slang_dict[k] = v
        else:
            slang_dict = raw_slang
    print(f"[Preprocess] Loaded {len(slang_dict)} slang normalization rules")

# Load Knowledge Base
df_kb = pd.DataFrame()
if os.path.exists(KB_PATH):
    df_kb = pd.read_csv(KB_PATH)
    print(f"[Knowledge Base] Loaded {len(df_kb)} beverage records from {KB_PATH}")

# -------------------------------------------------------------
# 2. Preprocessing & Entity Extraction (Dual-Script NER)
# -------------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    words = t.split()
    expanded = [slang_dict.get(w, w) for w in words]
    t = " ".join(expanded)
    t = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_slots(text: str) -> Dict[str, str]:
    t_low = text.lower()
    slots = {}
    
    # 1. Base Drinks
    if (
        "柠檬茶" in text or "檸檬茶" in text or
        re.search(r"\b(?:teh\s*o?\s*(?:ais|ice|peng|panas)?\s*limau|teh\s*limau|lemon\s*tea|limau\s*(?:tea|teh))\b", t_low)
    ):
        slots["base_drink"] = "Teh O Limau"
        slots["milk_type"] = "None"
    elif "鸳鸯" in text or "鴛鴦" in text or re.search(r"\b(?:cham|yuan yang|yuanyang)\b", t_low):
        slots["base_drink"] = "Cham"
    elif "白咖啡" in text or re.search(r"\b(?:white coffee)\b", t_low):
        slots["base_drink"] = "White Coffee"
    elif re.search(r"\bneslo\b", t_low):
        slots["base_drink"] = "Neslo"
    elif "美禄" in text or "美祿" in text or re.search(r"\b(?:milo|tak kiu|tut kiu)\b", t_low):
        slots["base_drink"] = "Milo"
    elif "美式" in text or re.search(r"\bamericano\b", t_low):
        slots["base_drink"] = "Americano"
        slots["milk_type"] = "None"
    elif "咖啡" in text or re.search(r"\b(?:kopi|coffee)\b", t_low):
        slots["base_drink"] = "Kopi"
    elif "奶茶" in text or "茶" in text or re.search(r"\b(?:teh|tea)\b", t_low):
        slots["base_drink"] = "Teh"

    # 2. Temperature
    if (
        "冰" in text or "冻" in text or "凍" in text or "冷" in text or
        re.search(r"\b(?:peng|ping|ice|iced|ais|sejuk|dingin|cold)\b", t_low)
    ):
        slots["temperature"] = "Iced"
    elif (
        "热" in text or "熱" in text or "温" in text or
        re.search(r"\b(?:panas|hot|suam|warm)\b", t_low)
    ):
        slots["temperature"] = "Hot"

    # 3. Sweetness
    if (
        "无糖" in text or "無糖" in text or "走糖" in text or
        re.search(r"\b(?:kosong|zero sugar|no sugar|tanpa gula|zero)\b|\b0\s*%", t_low)
    ):
        slots["sweetness_level"] = "Zero Sugar"
    elif (
        "少糖" in text or "微糖" in text or "少甜" in text or "半糖" in text or
        re.search(r"\b(?:siu dai|siew dai|less sugar|kurang manis|sedikit|half sugar|less sweet)\b|\b(?:50|30|25)\s*%", t_low)
    ):
        slots["sweetness_level"] = "Less Sugar"
    elif (
        "多糖" in text or "加糖" in text or "多甜" in text or
        re.search(r"\b(?:gah dai|ga dai|extra sweet|lebih manis|banyak|extra sugar)\b|\b120\s*%", t_low)
    ):
        slots["sweetness_level"] = "Extra Sugar"
    elif (
        "正常糖" in text or "普通糖" in text or "标准糖" in text or
        re.search(r"\b(?:normal sugar|biasa manis|manis biasa|std sugar)\b|\b100\s*%", t_low)
    ):
        slots["sweetness_level"] = "Normal"

    # 4. Milk Type
    if (
        "淡奶" in text or "炼奶" in text or "花奶" in text or
        re.search(r"\b(?:kopi c|teh c|cham c|neslo c)\b|\b(?:c|si|evaporated|susu sejat)\b", t_low)
    ):
        slots["milk_type"] = "Evaporated Milk"
    elif (
        "乌" in text or "黑咖啡" in text or "不加奶" in text or "不要奶" in text or "不用奶" in text or
        re.search(r"\b(?:kopi o|teh o|cham o|milo o|kopi all|teh all|milo all)\b|\b(?:o|kosong milk|no milk|tanpa susu|black|tak perlu|tak payah|tak nak susu|tak mau susu)\b", t_low) or
        (re.search(r"\bkosong\b", t_low) and not re.search(r"\b(?:susu|condensed|evaporated|c|tarik|biasa)\b", t_low))
    ):
        slots["milk_type"] = "None"
    elif (
        "奶" in text or
        re.search(r"\b(?:condensed|susu pekat|tarik|milk|susu)\b", t_low)
    ):
        slots["milk_type"] = "Condensed Milk"

    # 5. Serving Type
    if (
        "打包" in text or "带走" in text or "外带" in text or
        re.search(r"\b(?:tapau|bungkus|takeaway|take away|take out|pack)\b", t_low)
    ):
        slots["serving_type"] = "Takeaway"
    elif (
        "堂食" in text or "这里喝" in text or "内用" in text or
        re.search(r"\b(?:dine in|minum sini|sini|makan sini)\b", t_low)
    ):
        slots["serving_type"] = "Dine In"

    return slots

# -------------------------------------------------------------
# 3. Request Schema
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    utterance: str
    session_id: Optional[str] = "default"
    state: Optional[Dict[str, Any]] = None

# In-memory session tracking
sessions: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "status": "IDLE",
            "base_drink": None,
            "temperature": None,
            "sweetness_level": None,
            "milk_type": None,
            "serving_type": "Dine In",
            "pending_slot": None,
            "history": []
        }
    return sessions[session_id]

# -------------------------------------------------------------
# 4. API Endpoints
# -------------------------------------------------------------
# Groq Whisper API Key configuration (loaded from environment or .env)
def get_groq_api_key():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        env_file = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_file):
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("GROQ_API_KEY="):
                            key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass
    return key

GROQ_API_KEY = get_groq_api_key()

class TTSRequest(BaseModel):
    text: str
    lang: Optional[str] = "en"
    voice: Optional[str] = None
    tone: Optional[str] = "female"

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model": "LogisticRegression (TF-IDF N-gram 1-2)",
        "dataset_rows": 850,
        "classes": list(classes),
        "kb_items": len(df_kb),
        "stt_engine": "Groq Whisper Large V3 Turbo",
        "tts_engine": "Microsoft Edge-TTS Neural"
    }

@app.post("/api/transcribe")
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    lang: Optional[str] = Form("ms"),
    model: Optional[str] = Form("whisper-large-v3")
):
    """
    Transcribes multilingual speech (Manglish/Rojak/Malay/Chinese/Cantonese/English)
    via Groq Whisper Large V3 or Turbo with regional Kopitiam vocabulary priming.
    """
    try:
        audio_bytes = await file.read()
        filename = file.filename or "recording.webm"
        mime_type = file.content_type or "audio/webm"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        files = {
            "file": (filename, audio_bytes, mime_type)
        }
        
        selected_model = model if model in ["whisper-large-v3", "whisper-large-v3-turbo"] else "whisper-large-v3"
        
        prompt = "Pesanan Kopitiam: Kopi, Teh C Peng, Cham, Siu Dai, Kosong, Panas, Kurang Manis, Tapau, Bungkus, Milo, Ais, Limau, Siew Dai, Satu Milo suam, Kopi O"
        if lang == "zh":
            prompt = "南洋咖啡店点单：咖啡，奶茶，鸳鸯，美禄，少甜，无糖，打包，冰，热，少冰"
        elif lang == "en":
            prompt = "Kopitiam drinks order: Kopi, Teh, Milo, Cham, less sugar, no sugar, iced, hot, takeaway"

        data = {
            "model": selected_model,
            "prompt": prompt,
            "temperature": "0.0"
        }
        
        # Constrain language to eliminate low-resource hallucinations (like Icelandic or Czech)
        if lang and lang != "auto":
            data["language"] = lang
        
        res = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
            timeout=20
        )
        if res.status_code == 200:
            result = res.json()
            text = result.get("text", "").strip()
            return {"text": text, "status": "success", "engine": f"Groq {selected_model}"}
        else:
            return JSONResponse(status_code=res.status_code, content={"error": res.text, "status": "error"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "status": "error"})

@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    """
    Synthesizes natural, human-sounding speech via Microsoft Edge Neural Voices.
    """
    text = req.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Text is empty"})
    
    # Strip brackets so speech flows naturally without reading out technical parentheticals
    clean_speech = re.sub(r"（[^）]*）|\([^)]*\)", "", text).replace("*", "").replace("#", "").strip()
    if not clean_speech:
        clean_speech = text

    # Intelligently select Malaysian/Chinese/English neural voices
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", clean_speech))
    has_malay = bool(re.search(r"\b(?:boss|sila|minum|panas|sejuk|siap|boleh|pesanan|maaf|mesin|cawan|satu|gula|susu|nak|tukar|bungkus|tapau|kurang|manis|terima|kasih|jom)\b", clean_speech.lower()))
    
    tone = (req.tone or "female").lower()
    if req.voice:
        voice = req.voice
    elif has_chinese:
        voice = "zh-CN-YunxiNeural" if tone == "male" else "zh-CN-XiaoxiaoNeural"
    elif has_malay:
        voice = "ms-MY-OsmanNeural" if tone == "male" else "ms-MY-YasminNeural"
    else:
        voice = "en-MY-OsmanNeural" if tone == "male" else "en-MY-YasminNeural"
        
    try:
        communicate = edge_tts.Communicate(clean_speech, voice, rate="+25%")
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return Response(content=bytes(audio_data), media_type="audio/mpeg", headers={"X-Voice-Used": voice})
    except Exception as e:
        # Fallback to Google Cloud Natural TTS streaming audio if edge-tts connection times out on cloud datacenter
        try:
            glang = "zh-CN" if has_chinese else ("ms" if has_malay else "en")
            g_url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={requests.utils.quote(clean_speech)}&tl={glang}&client=tw-ob"
            g_res = requests.get(g_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if g_res.status_code == 200:
                return Response(content=g_res.content, media_type="audio/mpeg", headers={"X-Voice-Used": f"Google-Natural-{glang}"})
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/tts")
async def tts_get_endpoint(text: str, voice: Optional[str] = None, tone: Optional[str] = "female"):
    return await tts_endpoint(TTSRequest(text=text, voice=voice, tone=tone))

@app.post("/api/predict_intent")
def predict_intent_endpoint(req: ChatRequest):
    c_text = clean_text(req.utterance)
    X_vec = tfidf_vec.transform([c_text])
    pred = intent_model.predict(X_vec)[0]
    probs = intent_model.predict_proba(X_vec)[0]
    
    prob_dict = {c: float(round(p, 4)) for c, p in zip(classes, probs)}
    conf = float(round(probs[list(classes).index(pred)], 4))
    
    return {
        "raw_text": req.utterance,
        "clean_text": c_text,
        "predicted_intent": pred,
        "confidence": conf,
        "probabilities": prob_dict,
        "slots": extract_slots(req.utterance)
    }

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    utterance = req.utterance.strip()
    session_id = req.session_id or "default"
    state = req.state or get_session(session_id)
    
    c_text = clean_text(utterance)
    X_vec = tfidf_vec.transform([c_text])
    
    # ML Model Prediction
    pred_intent = intent_model.predict(X_vec)[0]
    probs = intent_model.predict_proba(X_vec)[0]
    prob_dict = {c: float(round(p, 4)) for c, p in zip(classes, probs)}
    conf = float(round(probs[list(classes).index(pred_intent)], 4))
    
    # Dual-script slot extraction
    extracted_slots = extract_slots(utterance)
    
    # Dialogue State Machine Execution
    is_zh = bool(re.search(r"[\u4e00-\u9fff]", utterance))
    t_low = utterance.lower()
    
    # Check physical vending machine single-cup limit
    if re.search(r"\b(?:tambah|tambah lagi|add another|two cups|2 cups|dua cawan)\b", t_low) or "再加一杯" in utterance or "两杯" in utterance:
        response = (
            "抱歉老板！ChatCoffee 无人咖啡机每次只能单杯出料（Single-Cup Dispenser Limit）。请先确认并取走当前这杯，再点下一杯哦！"
            if is_zh else
            "Maaf boss! Mesin ChatCoffee ini cuma boleh bancuh 1 cawan pada satu masa (Single-Cup Dispenser Limit). Sila sahkan & ambil cawan ini dulu, lepas tu baru order cawan seterusnya ya!"
        )
        return {
            "response": response,
            "intent": "single_cup_limit",
            "model_intent": pred_intent,
            "confidence": conf,
            "probabilities": prob_dict,
            "clean_text": c_text,
            "slots": extracted_slots,
            "state": state
        }
    
    # Reset command
    if utterance.lower() in ["reset", "clear", "semula", "batal", "cancel", "重置", "取消"]:
        state.update({
            "status": "IDLE", "base_drink": None, "temperature": None,
            "sweetness_level": None, "milk_type": None, "pending_slot": None
        })
        response = "好的，已重置订单！请问今天想喝什么？" if is_zh else "Pesanan dikosongkan. Jom order semula boss! Nak minum apa hari ini?"
        return {
            "response": response,
            "intent": "system_reset",
            "model_intent": pred_intent,
            "confidence": conf,
            "probabilities": prob_dict,
            "clean_text": c_text,
            "slots": {},
            "state": state
        }
    
    # Pending slot fill priority
    if state.get("pending_slot") and state["pending_slot"] in extracted_slots:
        slot_name = state["pending_slot"]
        state[slot_name] = extracted_slots[slot_name]
        state["pending_slot"] = None
    else:
        # Merge extracted slots
        for k, v in extracted_slots.items():
            state[k] = v

    # Health RAG query via unified_coffee_kb.csv
    if pred_intent == "health_query":
        # Search low calorie drinks from KB
        response = (
            "健康推荐：Kopi O Kosong Peng（仅 24 卡路里，0 糖分，RM 4.00）或 Iced Americano（仅 10 卡路里，RM 5.80）！需要为您来一杯吗？"
            if is_zh else
            "Untuk pilihan sihat & rendah kalori, kami cadangkan Kopi O Kosong Peng (hanya 24 kcal, 0g gula, RM 4.00) atau Iced Americano (10 kcal, RM 5.80)! Nak saya siapkan sekarang?"
        )
        return {
            "response": response,
            "intent": "health_query",
            "model_intent": pred_intent,
            "confidence": conf,
            "probabilities": prob_dict,
            "clean_text": c_text,
            "slots": extracted_slots,
            "state": state
        }
        
    # Order Modification
    if pred_intent == "modify_order":
        if "base_drink" in extracted_slots:
            state["base_drink"] = extracted_slots["base_drink"]
        if "temperature" in extracted_slots:
            state["temperature"] = extracted_slots["temperature"]
        if "sweetness_level" in extracted_slots:
            sweet = extracted_slots["sweetness_level"]
            condensed_drinks = ["Cham", "Kopi", "Teh", "White Coffee", "Milo", "Neslo"]
            is_condensed = state.get("milk_type") == "Condensed Milk" or (state.get("base_drink") in condensed_drinks and state.get("milk_type") not in ["None", "Evaporated Milk"])
            if sweet == "Zero Sugar" and is_condensed:
                state["sweetness_level"] = "Less Sugar"
            else:
                state["sweetness_level"] = sweet

    # Check missing slots
    if not state.get("base_drink"):
        response = "请问您想喝什么饮料呢？我们有 Kopi、Teh、Cham、Milo、White Coffee 等。" if is_zh else "Nak minum apa boss? Kita ada Kopi, Teh, Cham, Milo, White Coffee dan lain-lain."
        state["pending_slot"] = "base_drink"
        state["status"] = "GATHERING_SLOTS"
    elif not state.get("temperature"):
        drink = state["base_drink"]
        response = f"您选了 {drink}。请问要热的（Hot）还是加冰（Iced / Peng）呢？" if is_zh else f"Bancuh {drink}. Nak panas atau sejuk / ais (Peng) boss?"
        state["pending_slot"] = "temperature"
        state["status"] = "GATHERING_SLOTS"
    elif not state.get("sweetness_level"):
        response = "甜度想要少甜（Siu Dai / Less Sugar）、无糖（Kosong）还是正常甜？" if is_zh else "Manis macam mana boss? Kurang manis (Siu Dai), Kosong (Zero Sugar), atau Normal?"
        state["pending_slot"] = "sweetness_level"
        state["status"] = "GATHERING_SLOTS"
    else:
        # All slots collected -> AWAITING_CONFIRMATION
        state["status"] = "AWAITING_CONFIRMATION"
        drink_name = f"{state['base_drink']} {state.get('temperature', '')}"
        sweet = state.get("sweetness_level", "Normal")
        milk = state.get("milk_type", "Normal")
        
        # Calculate price from KB
        base_price = 4.00
        if state.get("temperature") == "Hot":
            base_price = 3.20
            
        response = (
            f"好的！一杯 {drink_name}（{sweet}，{milk}）。总共 RM {base_price:.2f}。请问确认下单吗？（请回答 OK / Confirm）"
            if is_zh else
            f"Beres boss! Satu {drink_name} ({sweet}, {milk}). Total RM {base_price:.2f}. Nak confirm order sekarang? (Jawab Confirm / Boleh)"
        )

    # If already awaiting confirmation and user confirms
    if state["status"] == "AWAITING_CONFIRMATION" and any(w in t_low for w in ["confirm", "ok", "yes", "ya", "boleh", "bayar", "好", "确认", "可以"]):
        state["status"] = "DISPENSING"
        response = (
            "订单已确认！正在为您现磨冲泡，请稍候从出料口取杯，谢谢老板！"
            if is_zh else
            "Order confirmed! Mesin sedang bancuh kopi anda. Sila ambil cawan di dispenser, terima kasih boss!"
        )

    return {
        "response": response,
        "intent": pred_intent,
        "model_name": "Logistic Regression (TF-IDF N-gram 1-2)",
        "confidence": conf,
        "probabilities": prob_dict,
        "clean_text": c_text,
        "slots": extracted_slots,
        "state": state
    }

# -------------------------------------------------------------
# 5. Serve Frontend Kiosk UI
# -------------------------------------------------------------
HTML_FILE = os.path.join(BASE_DIR, "index.html")
if not os.path.exists(HTML_FILE):
    HTML_FILE = os.path.join(BASE_DIR, "chatcoffee_malaysian_unmanned_smart_kiosk.html")

@app.get("/", response_class=HTMLResponse)
def serve_kiosk():
    if os.path.exists(HTML_FILE):
        with open(HTML_FILE, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>ChatCoffee Kiosk HTML not found</h1>")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting RojakBrew ML Kiosk on http://localhost:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
