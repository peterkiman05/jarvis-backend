from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, requests
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []

@app.get("/")
def home():
    return {"status": "JARVIS Signal Core Online"}

@app.get("/market/gold/analytics")
def get_gold_analytics():
    """Fetches XAU/USD data and computes technicals with 0.10 lot trade setup."""
    try:
        gold = yf.Ticker("XAUUSD=X")
        df = gold.history(period="5d", interval="15m")
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

        # Signal logic & parameters
        if latest_price >= ema_20:
            bias = "BULLISH (BUY SETUP)"
            entry = latest_price
            sl = round(latest_price - 15.00, 2)  # $15 Risk
            tp = round(latest_price + 30.00, 2)  # $30 Reward (1:2 R:R)
            sl_distance = round(entry - sl, 2)
            tp_distance = round(tp - entry, 2)
        else:
            bias = "BEARISH (SELL SETUP)"
            entry = latest_price
            sl = round(latest_price + 15.00, 2)  # $15 Risk
            tp = round(latest_price - 30.00, 2)  # $30 Reward (1:2 R:R)
            sl_distance = round(sl - entry, 2)
            tp_distance = round(entry - tp, 2)

        # 0.10 Lot calculations (10 oz contract: $1 move = $10 PnL)
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
        "You are JARVIS, an elite quantitative analyst and trading desk assistant. "
        "Provide exact entry, Stop Loss, Take Profit, and monetary risk/reward "
        "amounts for 0.10 lot positions on XAU/USD."
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

        # Calculate Technical Indicators
        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        ema_20 = EMAIndicator(close=df["Close"], window=20).ema_indicator().iloc[-1]
        ema_50 = EMAIndicator(close=df["Close"], window=50).ema_indicator().iloc[-1]

        latest_price = round(df["Close"].iloc[-1], 2)
        day_open = round(df["Open"].iloc[0], 2)
        high_price = round(df["High"].max(), 2)
        low_price = round(df["Low"].min(), 2)
        change = round(latest_price - day_open, 2)
        pct_change = round((change / day_open) * 100, 2)

        # Market condition interpretation
        trend = "BULLISH" if latest_price > ema_20 > ema_50 else ("BEARISH" if latest_price < ema_20 < ema_50 else "NEUTRAL / SIDEWAYS")
        rsi_signal = "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else "NEUTRAL")

        return {
            "symbol": "XAU/USD",
            "price": latest_price,
            "open": day_open,
            "high": high_price,
            "low": low_price,
            "change": change,
            "pct_change": f"{pct_change}%",
            "rsi_14": round(rsi, 2),
            "rsi_signal": rsi_signal,
            "ema_20": round(ema_20, 2),
            "ema_50": round(ema_50, 2),
            "bias": trend
        }
    except Exception as e:
        return {"error": f"Failed to compute technical analysis: {str(e)}"}

@app.post("/chat")
def chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")

    user_query = payload.message.lower()
    analytics_context = ""

    # Inject full technical analysis if gold/market is mentioned
    if any(k in user_query for k in ["gold", "xau", "tradingview", "market", "analysis", "rsi", "ema", "trend"]):
        data = get_gold_analytics()
        if "price" in data:
            analytics_context = (
                f"\n[REAL-TIME XAU/USD TECHNICAL ANALYTICS]\n"
                f"Current Price: ${data['price']} (Day Open: ${data['open']} | High: ${data['high']} | Low: ${data['low']})\n"
                f"Change: {data['pct_change']}\n"
                f"RSI (14): {data['rsi_14']} ({data['rsi_signal']})\n"
                f"EMA 20: ${data['ema_20']} | EMA 50: ${data['ema_50']}\n"
                f"Market Technical Bias: {data['bias']}"
            )

    system_prompt = (
        "You are JARVIS, an elite quantitative financial analyst and AI assistant. "
        "When asked about gold or trading, use the provided technical analytics (RSI, EMAs, Bias) "
        "to deliver sharp, professional market updates tailored for quick reading or voice output."
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
