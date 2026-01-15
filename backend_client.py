from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp


logger = logging.getLogger(__name__)


class BackendClient:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        session: aiohttp.ClientSession,
        timeout_seconds: float,
        retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._session = session
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._retries = retries

    async def send_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/ingest"
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        attempt = 0
        last_error: Exception | None = None
        while attempt <= self._retries:
            try:
                async with self._session.post(
                    url, json=payload, headers=headers, timeout=self._timeout
                ) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"backend error {resp.status}: {text}")
                    try:
                        return await resp.json()
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(f"invalid backend JSON: {text}") from exc
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
                last_error = exc
                attempt += 1
                logger.warning("backend request failed (attempt %s): %s", attempt, exc)
                await asyncio.sleep(0.2 * attempt)

        assert last_error is not None
        raise last_error
