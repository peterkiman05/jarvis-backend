from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, requests, urllib.parse, base64, sqlite3, json, io, sys
import pandas as pd
import yfinance as yf
from gtts import gTTS
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

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

ASSET_MAP = {
    "gold": "GC=F", "xau": "GC=F",
    "btc": "BTC-USD", "bitcoin": "BTC-USD",
    "eurusd": "EURUSD=X", "forex": "EURUSD=X",
    "us30": "^DJI", "dow": "^DJI"
}

DB_FILE = "kiemaen_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    cursor.execute("CREATE TABLE IF NOT EXISTS project_states (project_name TEXT PRIMARY KEY, blueprint TEXT, reference TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('default_lot', '0.10')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('risk_reward', '1:2')")
    conn.commit()
    conn.close()

init_db()

PENDING_TRADES: List[Dict[str, Any]] = []

class ChatRequest(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str

class TradeExecutionRequest(BaseModel):
    symbol: str
    action: str
    lot_size: float = 0.10
    stop_loss: float
    take_profit: float

class InventionProjectRequest(BaseModel):
    project_name: str
    description: str
    reference_project: Optional[str] = None

class CodeExecRequest(BaseModel):
    code: str

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

def fetch_recent_history(limit: int = 8) -> List[Dict[str, str]]:
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM history ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except:
        return []

def get_project_blueprint(project_name: str) -> Optional[str]:
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT blueprint FROM project_states WHERE project_name=?", (project_name,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except:
        return None

def save_project_blueprint(project_name: str, blueprint: str, reference: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO project_states (project_name, blueprint, reference) VALUES (?, ?, ?)", (project_name, blueprint, reference))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Project State Error: {e}")

@app.get("/")
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html", media_type="text/html")
    return {"status": "Kiemaen Production Omni Engine Online"}

@app.delete("/memory/reset")
def reset_memory():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return {"status": "SUCCESS", "message": "Conversation memory reset."}

@app.post("/trade/execute")
def execute_trade(trade: TradeExecutionRequest):
    order_payload = {
        "symbol": trade.symbol,
        "action": trade.action.upper(),
        "lot_size": trade.lot_size,
        "stop_loss": trade.stop_loss,
        "take_profit": trade.take_profit,
        "status": "PENDING"
    }
    PENDING_TRADES.append(order_payload)
    return {"status": "ORDER_QUEUED", "details": order_payload}

@app.get("/trade/pending")
def get_pending_trades():
    global PENDING_TRADES
    orders = PENDING_TRADES.copy()
    PENDING_TRADES.clear()
    return {"orders": orders}

@app.post("/execute-code")
def execute_python_code(payload: CodeExecRequest):
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
        exec_globals = {"pd": pd, "yf": yf, "math": __import__("math")}
        exec(payload.code, exec_globals)
        sys.stdout = sys.__stdout__
        output_val = buffer.getvalue()
        return {"output": output_val if output_val else "Code executed successfully with no printed output."}
    except Exception as e:
        sys.stdout = sys.__stdout__
        return {"error": str(e)}

@app.post("/tts/speak")
def speak_audio(payload: TTSRequest):
    try:
        clean_text = payload.text.replace("\n", ". ")
        if len(clean_text) > 800:
            clean_text = clean_text[:800] + "... output continued on screen."

        tts = gTTS(text=clean_text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(fp, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/invention/evolve")
def evolve_invention(payload: InventionProjectRequest):
    reference_data = ""
    if payload.reference_project:
        existing = get_project_blueprint(payload.reference_project)
        if existing:
            reference_data = f"\n[REFERENCE PROJECT BLUEPRINT ({payload.reference_project})]:\n{existing}\n"

    design_prompt = (
        f"You are Kiemaen's Advanced R&D and Invention Engine. "
        f"New Project Name: {payload.project_name}. Description: {payload.description}. "
        f"{reference_data}"
        f"Provide a comprehensive technical blueprint, component breakdown, hardware/software stack, and step-by-step integration guide building directly upon the reference design."
    )
    
    res = chat(ChatRequest(message=design_prompt))
    reply_text = res.get("reply", "")
    save_project_blueprint(payload.project_name, reply_text, payload.reference_project or "None")
    return {"reply": reply_text}

@app.get("/market/analytics/multi")
def get_multi_timeframe_analytics(asset: str = "gold"):
    try:
        symbol_key = asset.lower().replace("/", "").replace(" ", "")
        ticker_symbol = ASSET_MAP.get(symbol_key, "GC=F")
        ticker = yf.Ticker(ticker_symbol)

        timeframes = {"15m": "5d", "1h": "1mo", "1d": "3mo"}
        results = {}
        bullish_count = 0
        bearish_count = 0

        for tf, period in timeframes.items():
            df = ticker.history(period=period, interval=tf)
            if df.empty or len(df) < 20:
                continue

            rsi = round(RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1], 2)
            ema_20 = round(EMAIndicator(close=df["Close"], window=20).ema_indicator().iloc[-1], 2)
            price = round(df["Close"].iloc[-1], 2)
            bias = "BULLISH" if price >= ema_20 else "BEARISH"

            if bias == "BULLISH":
                bullish_count += 1
            else:
                bearish_count += 1

            results[tf] = {"price": price, "rsi": rsi, "ema_20": ema_20, "bias": bias}

        overall_bias = "STRONG BULLISH" if bullish_count > bearish_count else "STRONG BEARISH"
        lot_size = float(get_db_setting("default_lot", "0.10"))
        latest_price = list(results.values())[0]["price"] if results else 4350.0

        sl = round(latest_price * 0.995, 2) if "BULLISH" in overall_bias else round(latest_price * 1.005, 2)
        tp = round(latest_price * 1.010, 2) if "BULLISH" in overall_bias else round(latest_price * 0.990, 2)

        return {
            "asset": asset.upper(),
            "overall_bias": overall_bias,
            "timeframe_breakdown": results,
            "recommended_trade": {
                "symbol": ticker_symbol,
                "action": "BUY" if "BULLISH" in overall_bias else "SELL",
                "lot_size": lot_size,
                "entry": latest_price,
                "stop_loss": sl,
                "take_profit": tp
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze-chart")
async def analyze_chart(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")

    file_bytes = await file.read()
    base64_image = base64.b64encode(file_bytes).decode("utf-8")
    mime_type = file.content_type or "image/jpeg"
    data_url = f"data:{mime_type};base64,{base64_image}"

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this technical trading chart or civil engineering mechanics diagram in detail. Provide exact metrics, support/resistance levels, or formula breakdowns."},
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
            save_chat_memory("user", "[Uploaded Diagram/Chart]")
            save_chat_memory("assistant", analysis)
            return {"analysis": analysis}
        return {"analysis": f"Vision API Error: {res_data}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")

    user_query = payload.message.lower()
    save_chat_memory("user", payload.message)

    if any(k in user_query for k in ["video", "animate", "movie", "clip", "generate video", "short story"]):
        if REPLICATE_API_TOKEN:
            try:
                import replicate
                os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
                output = replicate.run("minimax/video-01", input={"prompt": payload.message, "prompt_optimizer": True})
                url = output.url if hasattr(output, "url") else str(output)
                reply = f"Generated cinematic video stream for your prompt:\n{url}"
                save_chat_memory("assistant", reply)
                return {"reply": reply}
            except Exception as e:
                print(f"Replicate Exception: {e}")

    if any(k in user_query for k in ["image", "picture", "photo", "draw", "visualize", "render"]):
        photo_prompt = f"A professional 8k rendering of {payload.message}, photorealistic, studio lighting, highly detailed"
        encoded = urllib.parse.quote(photo_prompt)
        img_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true"
        reply = f"Photorealistic rendering created:\n{img_url}"
        save_chat_memory("assistant", reply)
        return {"reply": reply}

    analytics_context = ""
    market_keywords = ["gold", "xau", "btc", "bitcoin", "eurusd", "us30", "dow", "market", "analysis", "trade", "buy", "sell"]
    
    if any(k in user_query for k in market_keywords):
        target_asset = "gold"
        if "btc" in user_query or "bitcoin" in user_query:
            target_asset = "btc"
        elif "eur" in user_query or "forex" in user_query:
            target_asset = "eurusd"
        elif "us30" in user_query or "dow" in user_query:
            target_asset = "us30"

        data = get_multi_timeframe_analytics(asset=target_asset)
        if "recommended_trade" in data:
            rec = data["recommended_trade"]
            analytics_context = (
                f"\n[LIVE MARKET DATA - {data['asset']}]\n"
                f"Overall Bias: {data['overall_bias']}\n"
                f"Timeframe breakdown: {json.dumps(data['timeframe_breakdown'])}\n"
                f"Trade Strategy ({rec['lot_size']} Lot):\n"
                f"• Setup: {rec['action']} @ ${rec['entry']}\n"
                f"• Stop Loss: ${rec['stop_loss']} | Take Profit: ${rec['take_profit']}\n"
            )

            if "execute" in user_query or "place trade" in user_query:
                execute_trade(TradeExecutionRequest(
                    symbol=rec["symbol"], action=rec["action"],
                    lot_size=rec["lot_size"], stop_loss=rec["stop_loss"], take_profit=rec["take_profit"]
                ))
                analytics_context += "\n[ACTION: ORDER DISPATCHED TO MT5 QUEUE]"

    lot_pref = get_db_setting("default_lot", "0.10")
    
    system_prompt = (
        f"You are Kiemaen, an elite general-purpose autonomous AI assistant designed for Jacob Peter Sithole. "
        f"Your active core domains: "
        f"1. Civil Engineering: University of Johannesburg modules, strength of materials, axial loading, normal stress, shear stress, strain calculations, and structural design. "
        f"2. Quantitative Trading: Gold XAUUSD, BTC, risk management, multi-timeframe RSI/EMA setups, lot size: {lot_pref}. "
        f"3. Content Creation: Scriptwriting, short animated storyboards, dynamic video/image prompts. "
        f"4. Invention R&D: Blueprinting new hardware and software tools using prior builds as foundational references. "
        f"Be direct, precise, mathematically sound, and articulate."
    )

    messages = [{"role": "system", "content": system_prompt + analytics_context}]
    for item in fetch_recent_history(limit=8):
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

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    file_bytes = await file.read()
    files = {"file": (file.filename or "recording.webm", file_bytes, file.content_type or "audio/webm")}
    data = {"model": "whisper-large-v3-turbo"}

    try:
        res = requests.post(GROQ_AUDIO_URL, headers=headers, files=files, data=data, timeout=15)
        res_data = res.json()
        if "text" not in res_data:
            raise HTTPException(status_code=500, detail=f"Groq Whisper Error: {res_data}")
        return {"text": res_data["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
