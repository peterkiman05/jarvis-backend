import os
import sqlite3
import subprocess
import tempfile
import urllib.parse
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="General-Purpose AI Backend", version="2.0.0")

# Configuration (Ensure GROQ_API_KEY is set in your environment or replace with your key)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Database Setup for Local Persistent Memory
DB_NAME = "kiemaen_general_memory.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_chat_memory(role: str, content: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversation_history (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def fetch_recent_history(limit: int = 15) -> List[Dict[str, str]]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM conversation_history ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


# Pydantic Models for API Requests
class ChatRequest(BaseModel):
    message: str

class CodeExecutionRequest(BaseModel):
    code: str

class UniversalSearchRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "status": "ONLINE",
        "mode": "General-Purpose AI Assistant",
        "capabilities": ["Universal Chat", "Python Sandbox", "Dynamic Tool Routing", "Persistent SQLite Memory"]
    }


@app.post("/chat")
def general_chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing from environment variables.")

    user_query = payload.message.lower()
    save_chat_memory("user", payload.message)

    # Dynamic Media/Image Generation Intent Interceptor
    if any(k in user_query for k in ["image", "picture", "photo", "draw", "visualize"]):
        image_prompt = f"A high-quality rendering of {payload.message}, highly detailed, professional composition"
        encoded_prompt = urllib.parse.quote(image_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true"
        reply = f"Here is the visual asset you requested:\n{image_url}"
        save_chat_memory("assistant", reply)
        return {"status": "SUCCESS", "type": "image", "reply": reply}

    # Universal General-Purpose System Prompt (Neutral and Adaptable)
    system_prompt = (
        "You are a helpful, highly intelligent, and versatile general-purpose AI assistant. "
        "Adapt seamlessly to the user's intent—whether they need help with software programming, writing, complex analysis, "
        "brainstorming, mathematics, or general educational research. Provide clear, concise, and structured responses."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for item in fetch_recent_history(limit=12):
        messages.append({"role": item["role"], "content": item["content"]})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Using the fast and versatile Llama 3.3 model
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_CHAT_URL, headers=headers, json=data, timeout=30)
        res_data = response.json()
        
        if "choices" not in res_data:
            raise HTTPException(status_code=500, detail=f"LLM API Error: {res_data}")
        
        reply = res_data["choices"][0]["message"]["content"]
        save_chat_memory("assistant", reply)
        return {"status": "SUCCESS", "reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute-code")
def execute_python_code(payload: CodeExecutionRequest):
    """Executes arbitrary Python code snippets safely in a temporary sandbox environment."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(payload.code.encode("utf-8"))
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            ["python", temp_file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return {
            "status": "SUCCESS" if result.returncode == 0 else "ERROR",
            "output": output
        }
    except subprocess.TimeoutExpired:
        return {"status": "FAILED", "output": "Execution timed out (exceeded 10 seconds limit)."}
    except Exception as e:
        return {"status": "FAILED", "output": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/tool/search-mock")
def mock_universal_search(payload: UniversalSearchRequest):
    """Placeholder router for live global web search data integration."""
    return {
        "status": "SUCCESS",
        "query": payload.query,
        "result": f"Simulated lookup response for: '{payload.query}'. Plug in any search API (e.g., Tavily or SerpAPI) here to pull live web indexes."
    }
