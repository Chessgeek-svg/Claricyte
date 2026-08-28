"""Generation backends behind one interface.

OpenAI serves the deployed demo; a local Hugging Face model runs the eval loop,
where the gold set gets replayed dozens of times while tuning and metered calls
would add up. Both answer the same messages so the eval can report both.

Imports are inside the classes so neither backend's dependency is paid unless it
is used, same reason as store.py.
"""

from __future__ import annotations

import os
from typing import Protocol

# Confirm against the account's own model list before relying on it; the cheap
# tier is renamed often. scripts/list_models.py prints what is actually available.
OPENAI_MODEL = "gpt-5-nano"
LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Answers are three or four sentences. The cap is a cost and abuse bound, not a
# style control; the prompt handles length.
MAX_OUTPUT_TOKENS = 400


class Provider(Protocol):
    """Anything that turns chat messages into an answer."""

    def generate(self, messages: list[dict[str, str]]) -> str: ...


def api_key() -> str:
    """The OpenAI key, from Streamlit secrets or the environment.

    Streamlit is optional so the eval scripts run outside the app.
    """
    try:
        import streamlit as st

        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "No OPENAI_API_KEY. Copy .streamlit/secrets.toml.example to "
            ".streamlit/secrets.toml and fill it in, or export the variable."
        )
    return key


class OpenAIProvider:
    """The hosted path. Cheap tier, capped output.

    temperature defaults to None, meaning "do not send it": the newer models
    reject the parameter outright. Set it to 0 once verified against the account's
    actual model, since the eval needs the same input to give the same output or a
    prompt change cannot be told apart from sampling noise.
    """

    def __init__(
        self,
        model: str = OPENAI_MODEL,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, messages: list[dict[str, str]]) -> str:
        from openai import OpenAI

        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = OpenAI(api_key=api_key()).chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_tokens,
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip()


class LocalProvider:
    """The eval path. Runs on the 2080 Super, free to call repeatedly."""

    def __init__(self, model: str = LOCAL_MODEL, max_tokens: int = MAX_OUTPUT_TOKENS):
        self.model = model
        self.max_tokens = max_tokens
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            self._pipe = pipeline(
                "text-generation",
                model=self.model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        return self._pipe

    def generate(self, messages: list[dict[str, str]]) -> str:
        out = self._pipeline()(
            messages, max_new_tokens=self.max_tokens, do_sample=False
        )
        return out[0]["generated_text"][-1]["content"].strip()


def get_provider(name: str = "openai") -> Provider:
    """Pick a backend by name."""
    if name == "openai":
        return OpenAIProvider()
    if name == "local":
        return LocalProvider()
    raise ValueError(f"unknown provider {name!r}; expected 'openai' or 'local'")
