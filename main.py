from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import os, requests
import yfinance as yf

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

@app.get("/")
def home():
    return {"status": "JARVIS Server Online"}

@app.get("/market/gold")
def get_gold_price():
    """Fetches real-time Gold vs USD (XAU/USD) market data."""
    try:
        gold = yf.Ticker("XAUUSD=X")
        data = gold.history(period="1d", interval="1m")
        if data.empty:
            gold = yf.Ticker("GC=F")  # Fallback to Gold Futures
            data = gold.history(period="1d", interval="1m")
        
        latest_price = round(data["Close"].iloc[-1], 2)
        day_open = round(data["Open"].iloc[0], 2)
        day_high = round(data["High"].max(), 2)
        day_low = round(data["Low"].min(), 2)
        change = round(latest_price - day_open, 2)
        pct_change = round((change / day_open) * 100, 2)

        return {
            "symbol": "XAU/USD",
            "price": latest_price,
            "open": day_open,
            "high": day_high,
            "low": day_low,
            "change": change,
            "pct_change": f"{pct_change}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market data fetch failed: {str(e)}")

@app.get("/generate-image")
def generate_image(prompt: str):
    """Generates an image URL using Pollinations AI."""
    encoded_prompt = requests.utils.quote(prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    return {"prompt": prompt, "image_url": image_url}

@app.post("/chat")
def chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")

    user_query = payload.message.lower()
    market_context = ""

    # Auto-inject real-time Gold market data if asked
    if any(k in user_query for k in ["gold", "xau", "tradingview", "market"]):
        try:
            m = get_gold_price()
            market_context = f"\n[REAL-TIME MARKET DATA: XAU/USD Price: ${m['price']} | Open: ${m['open']} | High: ${m['high']} | Low: ${m['low']} | Change: {m['pct_change']}]"
        except Exception:
            pass

    system_prompt = (
        "You are JARVIS, an advanced voice AI assistant and financial market analyst. "
        "Keep responses brief, sharp, and natural for voice playback."
    )

    messages = [{"role": "system", "content": system_prompt + market_context}]

    if payload.history:
        for msg in payload.history:
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": payload.message})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages
    }

    try:
        res = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
        res_data = res.json()
        if "choices" not in res_data:
            raise HTTPException(status_code=500, detail=f"Groq API Error: {res_data}")

        return {"reply": res_data["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    messages.append({"role": "user", "content": payload.message})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages
    }

    try:
        res = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
        res_data = res.json()
        if "choices" not in res_data:
            raise HTTPException(status_code=500, detail=f"Groq API Error: {res_data}")

        return {"reply": res_data["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    
    file_bytes = await file.read()
    files = {"file": (file.filename, file_bytes, file.content_type)}
    data = {"model": "whisper-large-v3-turbo"}

    try:
        res = requests.post(GROQ_AUDIO_URL, headers=headers, files=files, data=data)
        res_data = res.json()
        
        if "text" not in res_data:
            raise HTTPException(status_code=500, detail=f"Groq Whisper Error: {res_data}")

        return {"text": res_data["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
