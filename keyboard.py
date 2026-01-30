from __future__ import annotations

from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def build_keyboard(raw_keyboard: Any | None) -> InlineKeyboardMarkup | None:
    if not raw_keyboard:
        return None

    rows: list[list[InlineKeyboardButton]] = []
    for row in raw_keyboard:
        buttons: list[InlineKeyboardButton] = []
        for item in row:
            text = str(item.get("text") or "").strip()
            callback_data = item.get("callback_data") or item.get("data") or ""
            if not text:
                continue
            buttons.append(InlineKeyboardButton(text=text, callback_data=callback_data))
        if buttons:
            rows.append(buttons)

    if not rows:
        return None

    return InlineKeyboardMarkup(inline_keyboard=rows)
