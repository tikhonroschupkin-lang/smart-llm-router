"""Thin async client for the Perplexity Agent (Responses) API."""
import asyncio
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx

API_URL = "https://api.perplexity.ai/v1/responses"


def load_env(path: str = ".env") -> None:
    p = Path(__file__).parent / path
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


@dataclass
class LLMResult:
    model: str
    text: str
    cost: float
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_s: float
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class PerplexityClient:
    def __init__(self, concurrency: int = 16, timeout: float = 180.0, max_retries: int = 6):
        self.key = os.environ["PERPLEXITY_API_KEY"]
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.http = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
            limits=httpx.Limits(max_connections=concurrency * 2),
        )

    async def close(self):
        await self.http.aclose()

    async def complete(self, model: str, prompt: str, instructions: Optional[str] = None,
                       max_output_tokens: int = 16000) -> LLMResult:
        body = {"model": model, "input": prompt, "max_output_tokens": max_output_tokens}
        if instructions:
            body["instructions"] = instructions
        last_err = None
        async with self.sem:
            for attempt in range(self.max_retries):
                t0 = time.time()
                try:
                    r = await self.http.post(API_URL, json=body)
                    latency = time.time() - t0
                    if r.status_code == 429 or r.status_code >= 500:
                        last_err = "HTTP %d: %s" % (r.status_code, r.text[:300])
                        await asyncio.sleep(min(60, 2 ** attempt + 1))
                        continue
                    data = r.json()
                    if r.status_code != 200 or data.get("error"):
                        return LLMResult(model, "", 0.0, 0, 0, 0, latency,
                                         error="HTTP %d: %s" % (r.status_code, str(data.get("error"))[:300]))
                    text = "".join(
                        c.get("text", "")
                        for item in data.get("output", []) if item.get("type") == "message"
                        for c in item.get("content", []) if c.get("type") == "output_text"
                    )
                    u = data.get("usage") or {}
                    return LLMResult(
                        model=model,
                        text=text.strip(),
                        cost=float(((u.get("cost") or {}).get("total_cost")) or 0.0),
                        input_tokens=int(u.get("input_tokens") or 0),
                        output_tokens=int(u.get("output_tokens") or 0),
                        reasoning_tokens=int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0),
                        latency_s=latency,
                        error=None if data.get("status") == "completed" else "status=%s" % data.get("status"),
                    )
                except (httpx.HTTPError, ValueError) as e:
                    last_err = "%s: %s" % (type(e).__name__, e)
                    await asyncio.sleep(min(60, 2 ** attempt + 1))
        return LLMResult(model, "", 0.0, 0, 0, 0, 0.0, error="retries exhausted: %s" % last_err)
