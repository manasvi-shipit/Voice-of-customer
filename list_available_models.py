"""
list_available_models.py

Quick diagnostic -- asks your Gemini account directly which models it's
actually allowed to use, instead of guessing from documentation (which
changes fast and is often inconsistent for brand-new accounts).

Run this once to find the correct model name, then we'll plug it into
classify_reviews.py.
"""

import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='your-key'")

client = genai.Client(api_key=api_key)

print("Models available to your account that support generateContent:\n")
for model in client.models.list():
    actions = getattr(model, "supported_actions", None) or []
    if "generateContent" in actions or not actions:
        print(f"  {model.name}")
