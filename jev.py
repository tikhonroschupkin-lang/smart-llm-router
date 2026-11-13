"""Async client for TypeSafe's Jev decision model (POST https://api.typesafe.ai/v1/systemone).

Jev returns typed answers with probabilities instead of generated text:
  noul   -> {"noul": P(yes)}
  choice -> {"choice": name, "confidence": .., "probabilities": {name: p}}
  score  -> {"score": .., "probabilities": {"0": p, ...}}
"""
import asyncio
import os
import time

import httpx

import llm  # noqa: F401  (loads .env)

API_URL = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_M_INPUT = 0.042  # USD; output tokens are free


class JevClient:
    def __init__(self, model: str = "jev-latest", concurrency: int = 16, timeout: float = 60.0, max_retries: int = 6):
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.http = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": "Bearer " + os.environ["TYPESAFE_API_KEY"], "Content-Type": "application/json"},
        )

    async def close(self):
        await self.http.aclose()

    async def ask(self, state, questions: dict) -> dict:
        """Returns {"answers": {...}, "model": .., "cost": .., "latency_s": .., "error": ..}."""
        body = {"model": self.model, "state": state, "questions": questions}
        last = None
        async with self.sem:
            for attempt in range(self.max_retries):
                t0 = time.time()
                try:
                    r = await self.http.post(API_URL, json=body)
                except httpx.HTTPError as e:
                    last = "%s: %s" % (type(e).__name__, e)
                    await asyncio.sleep(min(30, 2 ** attempt))
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    last = "HTTP %d: %s" % (r.status_code, r.text[:200])
                    wait = r.headers.get("retry-after")
                    await asyncio.sleep(float(wait) if wait and wait.replace(".", "").isdigit() else min(30, 2 ** attempt))
                    continue
                if r.status_code != 200:
                    return {"answers": {}, "cost": 0.0, "latency_s": time.time() - t0,
                            "error": "HTTP %d: %s" % (r.status_code, r.text[:300])}
                d = r.json()
                tokens = (d.get("usage") or {}).get("input_tokens") or 0
                return {"answers": d.get("answers", {}), "model": d.get("model"),
                        "cost": tokens * PRICE_PER_M_INPUT / 1e6, "latency_s": time.time() - t0, "error": None}
        return {"answers": {}, "cost": 0.0, "latency_s": 0.0, "error": "retries exhausted: %s" % last}
