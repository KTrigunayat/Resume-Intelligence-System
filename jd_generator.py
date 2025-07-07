import os
import logging
from typing import List, Optional
import google.generativeai as genai

# --- API Key Configuration ---
# Primary: Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCb5B_wsF-91WRGlyd7N9C6DKqV37y3m-o")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Fallback: OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        pass  # OpenAI fallback is disabled if not installed

logger = logging.getLogger(__name__)

class JobDescriptionGenerator:
    """AI Agent for generating job descriptions using Gemini, fallback to OpenAI."""
    def __init__(self):
        pass

    def generate_jd(self, role_title: str, responsibilities: List[str], requirements: List[str], company: Optional[str] = None, perks: Optional[List[str]] = None) -> str:
        prompt_parts = [
            f"Generate a clear, inclusive, and compelling job description for the role titled '{role_title}'{f' at {company}' if company else ''}.",
            "\n\nKey Responsibilities:",
            "- " + "\n- ".join(responsibilities),
            "\n\nRequirements:",
            "- " + "\n- ".join(requirements)
        ]
        if perks:
            prompt_parts.append("\n\nPerks & Benefits:")
            prompt_parts.append("- " + "\n- ".join(perks))
        
        prompt_parts.append("\n\nFormat with clear sections: About the Role, Responsibilities, Requirements, and Perks/Benefits (if provided). Use a professional and engaging tone.")
        prompt = "\n".join(prompt_parts)

        # Try Gemini first
        if GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as gemini_e:
                logger.warning(f"JD generation with Gemini failed: {gemini_e}. Falling back to OpenAI.")
        
        # Fallback to OpenAI
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert HR professional and job description writer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=700,
                    temperature=0.5
                )
                return response.choices[0].message.content.strip()
            except Exception as openai_e:
                logger.error(f"JD generation with OpenAI failed: {openai_e}")

        return "[Error generating job description. Please check API configurations.]" 