from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, requests

app = FastAPI(title="Kiemaen Universal API")

# Enable CORS so the web UI frontend can communicate smoothly with your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "llama-3.3-70b-versatile"
    messages: list[Message]

@app.get("/v1/models")
def list_models():
    """Allows Open WebUI to automatically detect your available models."""
    return {
        "object": "list",
        "data": [{"id": "llama-3.3-70b-versatile", "object": "model", "owned_by": "kiemaen"}]
    }

@app.post("/v1/chat/completions")
def chat_completions(payload: ChatCompletionRequest):
    """OpenAI-compatible endpoint that Open WebUI connects to directly."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing.")

    formatted_messages = [{"role": m.role, "content": m.content} for m in payload.messages]

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": formatted_messages,
        "temperature": 0.7
    }

    try:
        res = requests.post(GROQ_CHAT_URL, headers=headers, json=data, timeout=30)
        res_data = res.json()
        
        # Format response back to standard OpenAI chat completion structure expected by WebUI
        reply_text = res_data["choices"][0]["message"]["content"]
        return {
            "id": "chatcmpl-kiemaen",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
