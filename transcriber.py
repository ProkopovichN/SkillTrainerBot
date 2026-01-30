from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import aiohttp
from aiogram import Bot
from aiogram.types import File, Voice

from config import Settings


logger = logging.getLogger(__name__)


class TranscriptionResult:
    def __init__(self, text: str, confidence: float | None = None):
        self.text = text
        self.confidence = confidence


class Transcriber:
    def __init__(
        self,
        settings: Settings,
        session: aiohttp.ClientSession,
    ) -> None:
        self.settings = settings
        self.session = session
        self.timeout = aiohttp.ClientTimeout(total=settings.asr_timeout_seconds)
        self._openrouter_enabled = bool(
            settings.openrouter_api_key and settings.openrouter_asr_model
        )
        self._openrouter_chat_fallback_model = settings.openrouter_asr_chat_model
        self._deepgram_enabled = bool(settings.deepgram_api_key)

    async def download_voice(self, bot: Bot, voice: Voice) -> Path:
        telegram_file: File = await bot.get_file(voice.file_id)
        suffix = Path(telegram_file.file_path or "").suffix or ".oga"
        fd, tmp_path = tempfile.mkstemp(prefix="voice_", suffix=suffix)
        os.close(fd)
        destination = Path(tmp_path)
        await bot.download_file(telegram_file.file_path, destination=destination)
        logger.debug("voice downloaded to %s", destination)
        return destination

    async def _convert_with_ffmpeg(self, source: Path) -> Path | None:
        ffmpeg_path = shutil.which(self.settings.ffmpeg_binary)
        if not ffmpeg_path:
            logger.warning("ffmpeg not found at '%s', sending original voice file", self.settings.ffmpeg_binary)
            return None
        logger.info("using ffmpeg at %s", ffmpeg_path)

        target = source.with_suffix(".wav")
        proc = await asyncio.create_subprocess_exec(
            ffmpeg_path,
            "-y",
            "-i",
            str(source),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(target),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.settings.asr_timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning("ffmpeg timed out, using original file")
            return None

        if proc.returncode != 0:
            err = ""
            try:
                err = (stderr or b"").decode("utf-8", "ignore")
            except Exception:
                err = ""
            logger.warning("ffmpeg failed with code %s stderr=%s", proc.returncode, err[:1000])
            return None

        return target

    async def transcribe(self, file_path: Path) -> TranscriptionResult:
        """
        Transcribe via Deepgram (if configured), else custom HTTP endpoint.
        """
        if self._deepgram_enabled:
            logger.info("transcriber: using Deepgram ASR url=%s", self.settings.deepgram_url)
            try:
                return await self._transcribe_deepgram(file_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("deepgram failed, fallback to next ASR: %s", exc)
        if self.settings.transcribe_url:
            logger.info("transcriber: using HTTP ASR endpoint %s", self.settings.transcribe_url)
            return await self._transcribe_http(file_path)
        raise RuntimeError("transcription is not configured")

    async def _transcribe_http(self, file_path: Path) -> TranscriptionResult:
        convert_path = await self._convert_with_ffmpeg(file_path)
        payload_path = convert_path or file_path
        headers: dict[str, str] = {}
        if self.settings.transcribe_token:
            headers["Authorization"] = f"Bearer {self.settings.transcribe_token}"

        data = aiohttp.FormData()
        file_handle = open(payload_path, "rb")
        data.add_field(
            "file",
            file_handle,
            filename=payload_path.name,
            content_type=_content_type(payload_path),
        )

        try:
            async with self.session.post(
                self.settings.transcribe_url,
                data=data,
                headers=headers,
                timeout=self.timeout,
            ) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"transcribe error {resp.status}: {body}")
                try:
                    result: dict[str, Any] = await resp.json()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"invalid transcribe JSON: {body}") from exc
        finally:
            file_handle.close()

        text = str(result.get("text") or "").strip()
        if not text:
            raise RuntimeError("transcriber returned empty text")

        confidence = _safe_float(result.get("confidence"))
        logger.info("transcriber: http ASR ok, len=%s, confidence=%s", len(text), confidence)
        return TranscriptionResult(text=text, confidence=confidence)

    async def _transcribe_deepgram(self, file_path: Path) -> TranscriptionResult:
        convert_path = await self._convert_with_ffmpeg(file_path)
        if not convert_path:
            # Deepgram может не понять OGG/Opus; требуем WAV.
            raise RuntimeError("ffmpeg not available, cannot transcode voice to WAV for Deepgram")
        payload_path = convert_path
        data = payload_path.read_bytes()
        headers = {
            "Authorization": f"Token {self.settings.deepgram_api_key}",
            "Content-Type": "audio/wav",
        }
        params = {"punctuate": "true", "smart_format": "true"}
        if self.settings.deepgram_model:
            params["model"] = self.settings.deepgram_model
        if self.settings.deepgram_language:
            params["language"] = self.settings.deepgram_language
        else:
            params["detect_language"] = "true"
        logger.info(
            "transcriber: deepgram payload bytes=%s content_type=%s params=%s",
            len(data),
            headers["Content-Type"],
            params,
        )
        async with self.session.post(
            self.settings.deepgram_url,
            data=data,
            headers=headers,
            params=params,
            timeout=self.timeout,
        ) as resp:
            raw = await resp.text()
            logger.info("transcriber: deepgram response status=%s preview=%s", resp.status, raw[:500])
            if resp.status >= 400:
                raise RuntimeError(f"deepgram error {resp.status}: {raw}")
            try:
                result: dict[str, Any] = await resp.json()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"invalid deepgram JSON: {raw}") from exc
        try:
            alt = result["results"]["channels"][0]["alternatives"][0]
            text = str(alt.get("transcript") or "").strip()
            confidence = _safe_float(alt.get("confidence"))
        except Exception:  # noqa: BLE001
            text = ""
            confidence = None
        if not text:
            raise RuntimeError(f"deepgram returned empty transcript raw={raw[:500]}")
        logger.info("transcriber: deepgram ASR ok, len=%s confidence=%s", len(text), confidence)
        return TranscriptionResult(text=text, confidence=confidence)

    async def _transcribe_openrouter_chat(
        self, file_path: Path, model_override: str | None = None
    ) -> TranscriptionResult:
        target_url = (
            self.settings.openrouter_asr_url.replace("audio/transcriptions", "chat/completions")
            if "audio/transcriptions" in self.settings.openrouter_asr_url
            else self.settings.openrouter_asr_url
        )
        convert_path = await self._convert_with_ffmpeg(file_path)
        payload_path = convert_path or Path(file_path)
        payload = payload_path.read_bytes()
        b64 = base64.b64encode(payload).decode("utf-8")
        fmt = _audio_format_from_suffix(payload_path.suffix)
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        model_to_use = model_override or self.settings.openrouter_asr_model
        body = {
            "model": model_to_use,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Расшифруй речь в текст. Отвечай только текстом."},
                        {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                    ],
                }
            ],
        }
        logger.info("transcriber: audio base64 payload=%s", b64)
        logger.info(body)
        async with self.session.post(target_url, json=body, headers=headers, timeout=self.timeout) as resp:
            raw = await resp.text()
            logger.info(
                "transcriber: chat ASR response status=%s preview=%s", resp.status, raw[:500]
            )
            if resp.status >= 400:
                raise RuntimeError(f"transcribe chat error {resp.status}: {raw}")
            try:
                data = await resp.json()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"invalid chat ASR JSON: {raw}") from exc

        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        text = None
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text") or "").strip()
                    if text:
                        break
        if not text:
            raise RuntimeError(f"chat ASR returned empty text: {content}")
        logger.info(
            "transcriber: openrouter chat ASR ok, len=%s, model=%s",
            len(text),
            model_to_use,
        )
        return TranscriptionResult(text=text, confidence=None)

    async def _transcribe_openrouter_chat_with_retry(self, file_path: Path) -> TranscriptionResult:
        # first try with primary model, then optional fallback chat model
        try_models = [self.settings.openrouter_asr_model]
        if self._openrouter_chat_fallback_model:
            try_models.append(self._openrouter_chat_fallback_model)
        last_error: Exception | None = None
        for m in try_models:
            try:
                return await self._transcribe_openrouter_chat(file_path, model_override=m)
            except Exception as exc:  # noqa: BLE001
                logger.warning("chat ASR with model %s failed: %s", m, exc)
                last_error = exc
                continue
        assert last_error is not None
        raise last_error


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".wav", ".wave"}:
        return "audio/wav"
    if suffix in {".ogg", ".oga"}:
        return "audio/ogg"
    if suffix in {".mp3"}:
        return "audio/mpeg"
    if suffix in {".m4a"}:
        return "audio/mp4"
    return "application/octet-stream"


def _audio_format_from_suffix(suffix: str) -> str:
    suf = (suffix or "").lower()
    if suf in {".wav", ".wave"}:
        return "wav"
    if suf in {".ogg", ".oga"}:
        return "ogg"
    if suf == ".mp3":
        return "mp3"
    if suf == ".m4a":
        return "m4a"
    return "wav"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None
