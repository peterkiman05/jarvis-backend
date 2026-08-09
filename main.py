from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, requests, urllib.parse, base64, sqlite3, json
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

ASSET_MAP = {
    "gold": "XAUUSD=X", "xau": "XAUUSD=X",
    "btc": "BTC-USD", "bitcoin": "BTC-USD",
    "eurusd": "EURUSD=X", "forex": "EURUSD=X",
    "us30": "^DJI", "dow": "^DJI"
}

# --- DATABASE SETUP (PERSISTENT MEMORY STORE) ---
DB_FILE = "jarvis_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('default_lot', '0.10')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('risk_reward', '1:2')")
    conn.commit()
    conn.close()

init_db()

class ChatRequest(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str

class SettingsRequest(BaseModel):
    default_lot: str
    risk_reward: str

class TradeExecutionRequest(BaseModel):
    symbol: str
    action: str
    lot_size: float = 0.10
    stop_loss: float
    take_profit: float

def get_db_setting(key: str, default: str) -> str:
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else default
    except:
        return default

def save_chat_memory(role: str, content: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO history (role, content) VALUES (?, ?)", (role, content))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Log Error: {e}")

def fetch_recent_history(limit: int = 6) -> List[Dict[str, str]]:
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM history ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except:
        return []

# --- 0. ROOT ROUTE ---
@app.get("/")
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html", media_type="text/html")
    return {"status": "JARVIS Core Online", "error": "index.html not found"}

# --- 1. MEMORY ROUTES ---
@app.post("/memory/settings")
def update_settings(payload: SettingsRequest):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('default_lot', ?)", (payload.default_lot,))
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('risk_reward', ?)", (payload.risk_reward,))
    conn.commit()
    conn.close()
    return {"status": "SUCCESS", "message": "Risk settings updated."}

@app.get("/memory/settings")
def get_settings():
    return {
        "default_lot": get_db_setting("default_lot", "0.10"),
        "risk_reward": get_db_setting("risk_reward", "1:2")
    }

@app.delete("/memory/reset")
def reset_memory():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return {"status": "SUCCESS", "message": "Conversation memory reset."}

# --- 2. ELEVENLABS & VISION ENDPOINTS ---
@app.post("/tts/speak")
def generate_speech(payload: TTSRequest):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=400, detail="ELEVENLABS_API_KEY missing.")

    clean_text = payload.text[:500]
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    data = {
        "text": clean_text,
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }

    try:
        res = requests.post(url, json=data, headers=headers, timeout=15)
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ElevenLabs Error: {res.text}")
        return Response(content=res.content, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-chart")
async def analyze_chart(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")

    file_bytes = await file.read()
    base64_image = base64.b64encode(file_bytes).decode("utf-8")
    mime_type = file.content_type or "image/jpeg"
    data_url = f"data:{mime_type};base64,{base64_image}"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this trading chart image. Identify key support, resistance, technical pattern, and clear trade recommendations."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ]
    }

    try:
        res = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=25)
        res_data = res.json()
        
        if "choices" in res_data:
            analysis = res_data["choices"][0]["message"]["content"]
            save_chat_memory("user", "[Uploaded Chart Image]")
            save_chat_memory("assistant", analysis)
            return {"analysis": analysis}

        return {"analysis": f"Vision API Error: {res_data}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. MULTI-ASSET QUANT ENGINE ---
@app.get("/market/analytics")
def get_market_analytics(asset: str = "gold"):
    try:
        symbol_key = asset.lower().replace("/", "").replace(" ", "")
        ticker_symbol = ASSET_MAP.get(symbol_key, "XAUUSD=X")
        
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="5d", interval="15m")

        if df.empty and ticker_symbol == "XAUUSD=X":
            ticker = yf.Ticker("GC=F")
            df = ticker.history(period="5d", interval="15m")

        if df.empty or len(df) < 50:
            return {"error": f"Insufficient data for asset: {asset}"}

        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        ema_20 = EMAIndicator(close=df["Close"], window=20).ema_indicator().iloc[-1]
        ema_50 = EMAIndicator(close=df["Close"], window=50).ema_indicator().iloc[-1]

        latest_price = round(df["Close"].iloc[-1], 2)
        day_open = round(df["Open"].iloc[0], 2)
        change = round(latest_price - day_open, 2)
        pct_change = round((change / day_open) * 100, 2)

        lot_size_setting = float(get_db_setting("default_lot", "0.10"))

        if latest_price >= ema_20:
            bias = "BULLISH (BUY SETUP)"
            entry = latest_price
            sl = round(latest_price * 0.995, 2)
            tp = round(latest_price * 1.010, 2)
        else:
            bias = "BEARISH (SELL SETUP)"
            entry = latest_price
            sl = round(latest_price * 1.005, 2)
            tp = round(latest_price * 0.990, 2)

        return {
            "asset": asset.upper(),
            "symbol": ticker_symbol,
            "price": latest_price,
            "change_pct": f"{pct_change}%",
            "rsi_14": round(rsi, 2),
            "ema_20": round(ema_20, 2),
            "ema_50": round(ema_50, 2),
            "bias": bias,
            "trade_setup": {
                "lot_size": lot_size_setting,
                "entry": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "risk_reward": get_db_setting("risk_reward", "1:2")
            }
        }
    except Exception as e:
        return {"error": f"Failed to compute trade setup: {str(e)}"}

# --- 4. MULTIMODAL CHAT ROUTER ---
@app.post("/chat")
def chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")

    user_query = payload.message.lower()
    save_chat_memory("user", payload.message)

    if any(k in user_query for k in ["image", "picture", "photo", "draw", "visualize", "render"]):
        photo_prompt = f"A professional 8k photograph of {payload.message}, shot on 35mm lens, realistic textures, studio lighting, photorealistic"
        encoded = urllib.parse.quote(photo_prompt)
        img_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true"
        reply = f"Here is your photorealistic rendering:\n{img_url}"
        save_chat_memory("assistant", reply)
        return {"reply": reply}

    analytics_context = ""
    market_keywords = ["gold", "xau", "btc", "bitcoin", "eurusd", "us30", "dow", "market", "analysis", "setup", "set up", "trade", "buy", "sell"]
    
    if any(k in user_query for k in market_keywords):
        target_asset = "gold"
        if "btc" in user_query or "bitcoin" in user_query:
            target_asset = "btc"
        elif "eur" in user_query or "forex" in user_query:
            target_asset = "eurusd"
        elif "us30" in user_query or "dow" in user_query:
            target_asset = "us30"

        data = get_market_analytics(asset=target_asset)
        if "price" in data and "trade_setup" in data:
            ts = data["trade_setup"]
            analytics_context = (
                f"\n[REAL-TIME MARKET SIGNAL - {data['asset']}]\n"
                f"Asset: {data['asset']} | Current Price: ${data['price']} | Bias: {data['bias']}\n"
                f"RSI (14): {data['rsi_14']} | EMA 20: ${data['ema_20']} | EMA 50: ${data['ema_50']}\n"
                f"Calculated {ts['lot_size']} Lot Position Parameters:\n"
                f"• Entry Level: ${ts['entry']}\n"
                f"• Stop Loss (SL): ${ts['stop_loss']}\n"
                f"• Take Profit (TP): ${ts['take_profit']}\n"
                f"• Risk-to-Reward: {ts['risk_reward']}\n"
            )

    lot_pref = get_db_setting("default_lot", "0.10")
    rr_pref = get_db_setting("risk_reward", "1:2")

    system_prompt = (
        f"You are JARVIS, an elite quantitative analyst. "
        f"User Preferences: Default Lot Size = {lot_pref}, Risk-to-Reward = {rr_pref}. "
        f"Be direct, sharp, and concise."
    )

    messages = [{"role": "system", "content": system_prompt + analytics_context}]
    history_logs = fetch_recent_history(limit=6)
    
    for item in history_logs:
        messages.append({"role": item["role"], "content": item["content"]})

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "llama-3.3-70b-versatile", "messages": messages}

    try:
        res = requests.post(GROQ_CHAT_URL, headers=headers, json=data, timeout=15)
        res_data = res.json()
        if "choices" not in res_data:
            raise HTTPException(status_code=500, detail=f"Groq API Error: {res_data}")

        reply = res_data["choices"][0]["message"]["content"]
        save_chat_memory("assistant", reply)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. WHISPER TRANSCRIPTION ---
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    file_bytes = await file.read()
    
    filename = file.filename or "recording.webm"
    content_type = file.content_type or "audio/webm"
    
    files = {"file": (filename, file_bytes, content_type)}
    data = {"model": "whisper-large-v3-turbo"}

    try:
        res = requests.post(GROQ_AUDIO_URL, headers=headers, files=files, data=data, timeout=15)
        res_data = res.json()
        if "text" not in res_data:
            raise HTTPException(status_code=500, detail=f"Groq Whisper Error: {res_data}")

        return {"text": res_data["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
