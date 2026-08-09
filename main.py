from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, requests, urllib.parse
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

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []

# --- 0. ROOT ROUTE (SERVES FILE DIRECTLY TO AVOID CSS PARSING ERRORS) ---
@app.get("/")
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html", media_type="text/html")
    return {"status": "JARVIS Core Online", "error": "index.html not found"}

# --- 1. STANDALONE PHOTOREALISTIC IMAGE ENDPOINT ---
@app.get("/generate-image")
def generate_image(prompt: str):
    photo_prompt = f"A professional 8k photograph of {prompt}, shot on 35mm lens, realistic textures, studio lighting, photorealistic, hyper-detailed, sharp focus"
    encoded = urllib.parse.quote(photo_prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true"
    return {"prompt": prompt, "image_url": image_url}

# --- 2. STANDALONE VIDEO ENDPOINT ---
@app.get("/generate-video")
def generate_video(prompt: str):
    if REPLICATE_API_TOKEN:
        try:
            import replicate
            output = replicate.run(
                "minimax/video-01",
                input={"prompt": prompt, "prompt_optimizer": True}
            )
            return {"prompt": prompt, "video_url": output}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Replicate Video Error: {str(e)}")
    
    encoded = urllib.parse.quote(f"cinematic video frame of {prompt}, 8k, photorealistic")
    return {
        "prompt": prompt,
        "note": "Set REPLICATE_API_TOKEN on Render for full MP4 renders.",
        "video_render_url": f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=576&model=flux&nologo=true"
    }

# --- 3. QUANTITATIVE XAU/USD SIGNAL ENGINE ---
@app.get("/market/gold/analytics")
def get_gold_analytics():
    try:
        gold = yf.Ticker("XAUUSD=X")
        df = gold.history(period="5d", interval="15m")
        if df.empty:
            gold = yf.Ticker("GC=F")
            df = gold.history(period="5d", interval="15m")

        if df.empty or len(df) < 50:
            return {"error": "Insufficient market data available for calculation."}

        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        ema_20 = EMAIndicator(close=df["Close"], window=20).ema_indicator().iloc[-1]
        ema_50 = EMAIndicator(close=df["Close"], window=50).ema_indicator().iloc[-1]

        latest_price = round(df["Close"].iloc[-1], 2)
        day_open = round(df["Open"].iloc[0], 2)
        high_price = round(df["High"].max(), 2)
        low_price = round(df["Low"].min(), 2)
        change = round(latest_price - day_open, 2)
        pct_change = round((change / day_open) * 100, 2)

        if latest_price >= ema_20:
            bias = "BULLISH (BUY SETUP)"
            entry = latest_price
            sl = round(latest_price - 15.00, 2)
            tp = round(latest_price + 30.00, 2)
        else:
            bias = "BEARISH (SELL SETUP)"
            entry = latest_price
            sl = round(latest_price + 15.00, 2)
            tp = round(latest_price - 30.00, 2)

        sl_distance = abs(entry - sl)
        tp_distance = abs(tp - entry)

        sl_usd = round(sl_distance * 10.0, 2)
        tp_usd = round(tp_distance * 10.0, 2)

        return {
            "symbol": "XAU/USD",
            "price": latest_price,
            "open": day_open,
            "high": high_price,
            "low": low_price,
            "change": change,
            "pct_change": f"{pct_change}%",
            "rsi_14": round(rsi, 2),
            "ema_20": round(ema_20, 2),
            "ema_50": round(ema_50, 2),
            "bias": bias,
            "trade_setup": {
                "lot_size": 0.10,
                "entry": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "sl_risk_usd": f"-${sl_usd}",
                "tp_reward_usd": f"+${tp_usd}",
                "risk_reward_ratio": "1:2"
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

    image_triggers = ["image", "picture", "photo", "draw", "visualize", "render", "ascii", "ingot"]
    if any(trigger in user_query for trigger in image_triggers):
        photo_prompt = f"A professional 8k photograph of {payload.message}, shot on 35mm lens, realistic textures, studio lighting, photorealistic"
        encoded = urllib.parse.quote(photo_prompt)
        img_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true"
        return {"reply": f"Here is your photorealistic rendering:\n{img_url}"}

    video_triggers = ["video", "animate", "movie", "clip"]
    if any(trigger in user_query for trigger in video_triggers):
        vid_res = generate_video(payload.message)
        url = vid_res.get("video_url") or vid_res.get("video_render_url")
        return {"reply": f"Here is your video generation link:\n{url}"}

    market_keywords = ["gold", "xau", "tradingview", "market", "analysis", "setup", "buy", "sell", "tp", "sl", "lot"]
    analytics_context = ""
    if any(k in user_query for k in market_keywords):
        data = get_gold_analytics()
        if "price" in data and "trade_setup" in data:
            ts = data["trade_setup"]
            analytics_context = (
                f"\n[REAL-TIME XAU/USD SIGNAL & TRADE SETUP]\n"
                f"Current Price: ${data['price']} | Bias: {data['bias']}\n"
                f"RSI (14): {data['rsi_14']} | EMA 20: ${data['ema_20']} | EMA 50: ${data['ema_50']}\n"
                f"Position Parameters (@ {ts['lot_size']} Lot Sizing):\n"
                f"• Entry Level: ${ts['entry']}\n"
                f"• Stop Loss (SL): ${ts['stop_loss']} (Max Loss: {ts['sl_risk_usd']})\n"
                f"• Take Profit (TP): ${ts['take_profit']} (Target Profit: {ts['tp_reward_usd']})\n"
                f"• Risk-to-Reward: {ts['risk_reward_ratio']}"
            )

    system_prompt = (
        "You are JARVIS, an elite quantitative analyst and AI assistant. "
        "Keep responses professional, sharp, direct, and concise."
    )

    messages = [{"role": "system", "content": system_prompt + analytics_context}]

    if payload.history:
        for item in payload.history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append({"role": str(item["role"]), "content": str(item["content"])})

    messages.append({"role": "user", "content": payload.message})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {"model": "llama-3.3-70b-versatile", "messages": messages}

    try:
        res = requests.post(GROQ_CHAT_URL, headers=headers, json=data, timeout=15)
        res_data = res.json()
        if "choices" not in res_data:
            raise HTTPException(status_code=500, detail=f"Groq API Error: {res_data}")

        return {"reply": res_data["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
