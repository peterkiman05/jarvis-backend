from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, requests, urllib.parse
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN") # Optional for Video Generation

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []

@app.get("/")
def home():
    return {"status": "JARVIS Multimodal & Quantitative Core Online"}

# --- 1. PHOTOREALISTIC IMAGE GENERATION ---
@app.get("/generate-image")
def generate_image(prompt: str):
    """Generates ultra-realistic 8k photos using Pollinations FLUX Engine."""
    # Force real photography styles into the prompt automatically
    photo_prompt = f"A professional 8k photograph of {prompt}, shot on 35mm lens, realistic textures, cinematic lighting, photorealistic, hyper-detailed, high quality"
    encoded = urllib.parse.quote(photo_prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true"
    return {"prompt": prompt, "image_url": image_url}

# --- 2. VIDEO GENERATION ROUTE ---
@app.get("/generate-video")
def generate_video(prompt: str):
    """Generates video clips using Replicate or Pollinations fallback."""
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
    
    # Free alternative fallback URL
    encoded = urllib.parse.quote(prompt)
    return {
        "prompt": prompt,
        "note": "For dedicated MP4 AI video generation, set REPLICATE_API_TOKEN on Render.",
        "video_render_url": f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=576&model=flux&nologo=true"
    }

# --- 3. QUANTITATIVE TRADING ENGINE ---
@app.get("/market/gold/analytics")
def get_gold_analytics():
    """Fetches live XAU/USD technicals with 0.10 lot position sizing."""
    try:
        gold = yf.Ticker("XAUUSD=X")
        df = gold.history(period="5d", interval="15m")
        if df.empty:
            gold = yf.Ticker("GC=F")
            df = gold.history(period="5d", interval="15m")

        if df.empty or len(df) < 50:
            return {"error": "Insufficient market data available."}

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
                "sl_risk_usd": f"-${round(sl_distance * 10.0, 2)}",
                "tp_reward_usd": f"+${round(tp_distance * 10.0, 2)}",
                "risk_reward_ratio": "1:2"
            }
        }
    except Exception as e:
        return {"error": f"Technical computation failed: {str(e)}"}

# --- 4. CHAT ROUTER ---
@app.post("/chat")
def chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")

    user_query = payload.message.lower()

    # Video Generation Request Interceptor
    if any(k in user_query for k in ["generate video", "make a video", "create video", "animate"]):
        vid_res = generate_video(payload.message)
        url = vid_res.get("video_url") or vid_res.get("video_render_url")
        return {"reply": f"Here is the video generation link for your query:\n{url}"}

    # Photorealistic Image Interceptor
    if any(k in user_query for k in ["generate image", "draw", "create photo", "picture of", "show photo"]):
        img_res = generate_image(payload.message)
        return {"reply": f"Here is your photorealistic rendering:\n{img_res['image_url']}"}

    # Financial Signal Context Injection
    analytics_context = ""
    if any(k in user_query for k in ["gold", "xau", "tradingview", "market", "analysis", "setup", "buy", "sell", "tp", "sl"]):
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
        "Keep responses brief, sharp, and practical."
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
        if df.empty:
            gold = yf.Ticker("GC=F")
            df = gold.history(period="5d", interval="15m")

        if df.empty or len(df) < 50:
            return {"error": "Insufficient market data available for setup."}

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
            sl_distance = round(entry - sl, 2)
            tp_distance = round(tp - entry, 2)
        else:
            bias = "BEARISH (SELL SETUP)"
            entry = latest_price
            sl = round(latest_price + 15.00, 2)
            tp = round(latest_price - 30.00, 2)
            sl_distance = round(sl - entry, 2)
            tp_distance = round(entry - tp, 2)

        sl_pnl_usd = round(sl_distance * 10.0, 2)
        tp_pnl_usd = round(tp_distance * 10.0, 2)

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
                "sl_risk_usd": f"-${sl_pnl_usd}",
                "tp_reward_usd": f"+${tp_pnl_usd}",
                "risk_reward_ratio": "1:2"
            }
        }
    except Exception as e:
        return {"error": f"Failed to calculate trade setup: {str(e)}"}

@app.post("/chat")
def chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")

    user_query = payload.message.lower()
    analytics_context = ""

    # Check if user wants an image generated
    if any(k in user_query for k in ["generate image", "draw", "create picture", "show chart image", "visualize"]):
        encoded_prompt = urllib.parse.quote(payload.message)
        img_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
        return {"reply": f"Here is the generated image for your request:\n{img_url}"}

    if any(k in user_query for k in ["gold", "xau", "tradingview", "market", "analysis", "setup", "buy", "sell", "tp", "sl", "lot"]):
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
        "Provide clear, concise, and actionable information."
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

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages
    }

    try:
        res = requests.post(GROQ_CHAT_URL, headers=headers, json=data, timeout=15)
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
        res = requests.post(GROQ_AUDIO_URL, headers=headers, files=files, data=data, timeout=15)
        res_data = res.json()
        if "text" not in res_data:
            raise HTTPException(status_code=500, detail=f"Groq Whisper Error: {res_data}")

        return {"text": res_data["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
