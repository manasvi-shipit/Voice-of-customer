"""
llm_provider.py

A single, pluggable interface for calling ANY LLM backend, so the rest of
the pipeline (classify_reviews.py, generate_citations.py, and their _v2
counterparts) never needs to know which provider is actually running
underneath.

SUPPORTED PROVIDERS:
    "gemini"           Google Gemini (the original backend this project
                        was built and evaluated on).
    "openai_compatible" Any endpoint that speaks the OpenAI chat-completions
                        API shape. This one adapter covers real OpenAI,
                        Ollama (local, free, no API key), Groq, OpenRouter,
                        LM Studio, vLLM, Together.ai, and most self-hosted
                        inference servers -- they're all wire-compatible.

CONFIGURATION (environment variables -- nothing to edit in code):
    LLM_PROVIDER   "gemini" or "openai_compatible"  (default: "gemini",
                   so existing setups keep working with zero changes)
    LLM_MODEL      model name, e.g. "gemini-3.5-flash-lite", "gpt-4o-mini",
                   "llama3.1", "meta-llama/llama-3.1-8b-instruct"
    LLM_API_KEY    API key. Required for Gemini and real OpenAI. NOT
                   required for most local servers (Ollama etc.) -- any
                   placeholder value is fine, or leave it unset.
    LLM_BASE_URL   only used for openai_compatible. Examples:
                       Ollama:      http://localhost:11434/v1
                       OpenRouter:  https://openrouter.ai/api/v1
                       LM Studio:   http://localhost:1234/v1
                       Real OpenAI: leave unset (defaults to api.openai.com)

BACKWARD COMPATIBILITY: if none of these are set, everything behaves
exactly as before -- provider defaults to "gemini", and GEMINI_API_KEY
(the original env var) is still read as a fallback for LLM_API_KEY.

USAGE:
    from llm_provider import call_llm, validate_config

    validate_config()  # fail fast with a clear message before doing any work
    response_text = call_llm("your prompt here")

SETUP:
    Gemini (original):        pip3 install google-genai
    Anything OpenAI-compatible: pip3 install openai
    (Only the one you're actually using needs to be installed.)
"""

import os


def _current_provider():
    return os.environ.get("LLM_PROVIDER", "gemini").strip().lower()


def validate_config():
    """
    Raises a clear, actionable RuntimeError if the current provider isn't
    properly configured -- called once up front so a script fails fast
    with a useful message instead of dying deep inside a retry loop.
    """
    provider = _current_provider()

    if provider == "gemini":
        if not (os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            raise RuntimeError(
                "GEMINI_API_KEY (or LLM_API_KEY) not set. "
                "Run: export GEMINI_API_KEY='your-key-here'"
            )

    elif provider == "openai_compatible":
        if not os.environ.get("LLM_MODEL"):
            raise RuntimeError(
                "LLM_MODEL not set. Examples: "
                "export LLM_MODEL='llama3.1' (Ollama), "
                "export LLM_MODEL='gpt-4o-mini' (OpenAI)."
            )
        base_url = os.environ.get("LLM_BASE_URL")
        if not base_url and not os.environ.get("LLM_API_KEY"):
            raise RuntimeError(
                "No LLM_API_KEY set and no LLM_BASE_URL set. For real OpenAI, "
                "set LLM_API_KEY. For a local server (e.g. Ollama), set "
                "LLM_BASE_URL='http://localhost:11434/v1' -- an API key isn't needed."
            )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini' or 'openai_compatible'."
        )


def call_llm(prompt: str) -> str:
    """Sends prompt to whichever provider is configured, returns the raw
    text response. Retry/backoff logic stays in the calling script (the
    existing classify_batch()/call_model_for_theme() loops already handle
    that) -- this function does exactly one call, and raises on failure."""
    provider = _current_provider()
    if provider == "gemini":
        return _call_gemini(prompt)
    elif provider == "openai_compatible":
        return _call_openai_compatible(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'.")


def _call_gemini(prompt: str) -> str:
    from google import genai

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or LLM_API_KEY) not set.")

    model = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def _call_openai_compatible(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("LLM_API_KEY", "not-needed")  # local servers ignore this
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL not set.")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
