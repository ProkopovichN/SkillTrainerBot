from __future__ import annotations

from typing import Iterable


def chunk_text(text: str, limit: int) -> Iterable[str]:
    """
    Split text into chunks respecting the limit.
    Prefers paragraph boundaries; falls back to hard splits if needed.
    """
    paragraphs = text.split("\n\n")
    current: list[str] = []
    current_len = 0

    def flush() -> str | None:
        nonlocal current, current_len
        if not current:
            return None
        chunk = "\n\n".join(current)
        current = []
        current_len = 0
        return chunk

    for para in paragraphs:
        para_len = len(para)
        # If paragraph itself exceeds limit, hard-split it.
        if para_len > limit:
            # flush what we have
            chunk = flush()
            if chunk:
                yield chunk
            start = 0
            while start < para_len:
                end = min(para_len, start + limit)
                yield para[start:end]
                start = end
            continue

        if current_len + (2 if current else 0) + para_len <= limit:
            current.append(para)
            current_len += (2 if current_len else 0) + para_len
        else:
            chunk = flush()
            if chunk:
                yield chunk
            current.append(para)
            current_len = para_len

    chunk = flush()
    if chunk:
        yield chunk
