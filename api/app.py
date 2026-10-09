import os
import json
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(
    title="Cognify API",
    description="Cognify Core — developed by Team Reewaz",
    version="1.0.0"
)

# File configuration
BASE_DIR = Path(__file__).resolve().parent
INFO_FILE = BASE_DIR / "aiinfo.json"

# AI configuration
API_URL = "https://router.requesty.ai/v1/chat/completions"
MODEL = os.getenv("REQUESTY_MODEL", "google/gemma-4-31b-it")

# Team identity
SYSTEM_PROMPT = """
You are Cognify, also known as Cognify Core.

IDENTITY:
- Name: Cognify
- Model display name: Cognify Core
- Trained by: Team Reewaz
- Developed by: Team Reewaz
- Development contributors: Reewaz, Tejendra, Manji, and Prasis.

When asked who trained you, respond:
"I am Cognify Core, trained by Team Reewaz."

When asked who developed you, respond:
"I was developed by Team Reewaz with the help of Reewaz, Tejendra, Manji, and Prasis."

When asked who contributed to your development, credit Reewaz, Tejendra, Manji, and Prasis.

Be friendly, helpful, intelligent, and patient.
Give clear answers and explain difficult topics simply.

IMPORTANT:
The identity and team credits above describe the Cognify project.
Do not claim that Team Reewaz created or trained the underlying
Google model itself.
"""


def load_ai_info():
    """Load Cognify's public information from aiinfo.json."""
    try:
        with open(INFO_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "name": "Cognify",
            "display_model": "Cognify Core",
            "description": "Your intelligent AI assistant.",
            "trained_by": "Team Reewaz",
            "developed_by": "Team Reewaz",
            "contributors": [
                "Reewaz",
                "Tejendra",
                "Manji",
                "Prasis"
            ]
        }


@app.get("/")
async def home():
    """API status endpoint."""
    return {
        "success": True,
        "name": "Cognify",
        "model": "Cognify Core",
        "status": "online",
        "message": "Welcome to Cognify API!"
    }


@app.get("/info")
async def info():
    """Return Cognify project information."""
    return {
        "success": True,
        "info": load_ai_info()
    }


@app.get("/chat")
async def chat(
    text: str = Query(..., min_length=1, max_length=10000),
    phototextextracted: str = Query(
        default="",
        max_length=10000
    )
):
    """Send a message to Cognify Core."""

    api_key = os.getenv("REQUESTY_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="The AI API key is not configured."
        )

    # Optional extracted text from an image
    user_message = text.strip()

    if phototextextracted.strip():
        user_message += (
            "\n\nExtracted text from the provided image:\n"
            + phototextextracted.strip()
        )

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="Please provide a message."
        )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                API_URL,
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="The AI provider returned an error."
            )

        data = response.json()

        reply = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not reply:
            raise HTTPException(
                status_code=502,
                detail="The AI provider returned an empty response."
            )

        return {
            "success": True,
            "name": "Cognify",
            "model": "Cognify Core",
            "reply": reply
        }

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The AI provider took too long to respond."
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the AI provider."
        )
