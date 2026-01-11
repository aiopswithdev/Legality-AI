import os
import json
import time
from google import genai
from pydantic import BaseModel
from typing import Literal
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-3-flash-preview"

# 1. Define the Schema strictly using Pydantic
class RiskAssessment(BaseModel):
    relevant: bool
    risk_strength: Literal["low", "medium", "high"]
    suggested_band: Literal["low", "review", "high"]
    confidence: float

SYSTEM_PROMPT = """You are a legal risk quality auditor.
You do not provide legal advice.
You judge whether a clause clearly represents the specified legal risk.
Be conservative and precise.
Respond strictly in valid JSON.
"""


def call_llm_gemini(clause_text: str, label: str) -> dict:
    prompt = f"""
Clause:
\"\"\"{clause_text}\"\"\"

Label:
{label}

Respond ONLY in JSON with this schema:
{{
  "relevant": true | false,
  "risk_strength": "low" | "medium" | "high",
  "suggested_band": "low" | "review" | "high",
  "confidence": number between 0 and 1
}}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                # 2. Pass System Prompt here for better adherence
                system_instruction=SYSTEM_PROMPT, 
                # 3. Enforce the Schema strictly
                response_mime_type="application/json",
                response_schema=RiskAssessment, 
            ),
        )

        # 4. Automatic Pydantic parsing (No json.loads needed)
        # The SDK automatically hydrates the object if schema is provided
        result: RiskAssessment = response.parsed 
        
        return result.model_dump()

    except Exception as e:
        print(f"❌ API Error: {e}")
        # Return a fallback "error" dict so your pipeline doesn't crash
        return {
            "relevant": False,
            "risk_strength": "error",
            "suggested_band": "review",
            "confidence": 0.0
        }