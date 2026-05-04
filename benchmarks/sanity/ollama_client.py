from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaResponse:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def call_ollama(
    prompt: str,
    model: str,
    base_url: str = "http://localhost:11434",
    timeout_s: float = 30.0,
) -> OllamaResponse:
    """Send a single prompt to Ollama and return the response."""
    url = f"{base_url}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode())
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return OllamaResponse(
            text=raw.get("response", ""),
            latency_ms=latency_ms,
            prompt_tokens=raw.get("prompt_eval_count", 0),
            completion_tokens=raw.get("eval_count", 0),
        )
    except (urllib.error.URLError, OSError) as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return OllamaResponse(
            text="",
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return OllamaResponse(
            text="",
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            error=f"Unexpected error: {exc}",
        )


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Return True if Ollama is reachable, False otherwise."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3.0):
            return True
    except Exception:  # noqa: BLE001
        return False
