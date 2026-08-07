frfrom fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import os, requests

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

@app.post("/chat")
def chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")

    messages = [
        {"role": "system", "content": "You are JARVIS, an advanced voice AI assistant. Keep responses brief, natural, and friendly for voice playback."}
    ]

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
