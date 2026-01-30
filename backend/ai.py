from __future__ import annotations

import random
from typing import Tuple

from config import Settings


def evaluate_answer(text: str, settings: Settings) -> Tuple[bool, str]:
    """
    Lightweight heuristic to simulate AI feedback.
    """
    lower = text.lower()
    score = 0
    for kw in settings.ai_positive_keywords:
        if kw in lower:
            score += 1
    score += int(len(text.split()) > 12)
    good = score >= 2
    if good:
        return True, random.choice(
            [
                "Отлично: есть конкретика и фокус на действия. Давай двигаться дальше.",
                "Хорошо сформулировано, видно рабочие шаги. Готов к следующему кейсу.",
            ]
        )
    return (
        False,
        "Ответ пока поверхностный. Добавь конкретики: примеры, действия, ожидания. Попробуем ещё раз?",
    )


def interpret_diagnostic(answers: list[str]) -> str:
    if not answers:
        return "Диагностика не заполнена."
    positives = sum(1 for a in answers if "сильн" in a.lower() or "ок" in a.lower())
    if positives >= len(answers) / 2:
        return "Уровень базовый/средний: есть сильные стороны, но стоит потренировать структурность."
    return "Диагностика показывает зоны роста: обратная связь пока размыта. Предлагаю начать с тренажёра и закрепить формат."
