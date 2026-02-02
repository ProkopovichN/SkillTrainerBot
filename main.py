from __future__ import annotations

from dataclasses import dataclass, field

import asyncio
import json
import logging
import random
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

import aiohttp
from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)

from backend_client import BackendClient
from config import Settings
from keyboard import build_keyboard
from transcriber import Transcriber
from utils import chunk_text


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
LOADING_MESSAGES = [
    "Думаю над ответом...",
    "Сверяюсь с тренажёром, секундочку.",
    "Перебираю лучшие подсказки для тебя.",
    "Считаю баллы, не переключайся!",
]


class UpdateDeduplicator:
    def __init__(self, max_size: int = 2048) -> None:
        self._seen: set[int] = set()
        self._queue: deque[int] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    async def is_duplicate(self, update_id: int) -> bool:
        async with self._lock:
            if update_id in self._seen:
                return True
            if len(self._queue) >= self._queue.maxlen:
                oldest = self._queue.popleft()
                self._seen.discard(oldest)
            self._queue.append(update_id)
            self._seen.add(update_id)
            return False


def minimal_raw_message(message: Message) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "date": message.date.isoformat(),
        "text": message.text or message.caption or None,
        "voice": {
            "file_id": message.voice.file_id,
            "duration": message.voice.duration,
        }
        if message.voice
        else None,
    }


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Начать диагностику", callback_data="action:diagnostic:start"
                ),
                InlineKeyboardButton(
                    text="Перейти к тренажеру", callback_data="action:training:start"
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Навык: обратная связь", callback_data="action:skill:feedback"
                ),
                InlineKeyboardButton(
                    text="Навык: ИПР", callback_data="action:skill:idp"
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Сфера деятельности", callback_data="action:sphere:menu"
                )
            ],
        ]
    )


EXPERIENCE_LEVEL_OPTIONS = [
    ("newbie", "Новичок (до года)"),
    ("1-3", "1-3 года"),
    ("3plus", "3+ лет"),
]
EXPERIENCE_LEVEL_LABELS = {key: label for key, label in EXPERIENCE_LEVEL_OPTIONS}

MEETING_OPTIONS = [
    ("results", "Результаты сотрудника"),
    ("leader", "Ожидания руководителя"),
    ("outcome", "Ожидания результата"),
]
MEETING_OPTION_KEYS = {key for key, _ in MEETING_OPTIONS}
MEETING_PROMPT = (
    "Какие встречи тебе предстоят? 🎯\n"
    "Выбери одну или несколько тем:"
)


def build_experience_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=EXPERIENCE_LEVEL_OPTIONS[0][1],
                    callback_data=f"action:experience:{EXPERIENCE_LEVEL_OPTIONS[0][0]}",
                )
            ],
            [
                InlineKeyboardButton(
                    text=EXPERIENCE_LEVEL_OPTIONS[1][1],
                    callback_data=f"action:experience:{EXPERIENCE_LEVEL_OPTIONS[1][0]}",
                ),
                InlineKeyboardButton(
                    text=EXPERIENCE_LEVEL_OPTIONS[2][1],
                    callback_data=f"action:experience:{EXPERIENCE_LEVEL_OPTIONS[2][0]}",
                ),
            ],
        ]
    )


@dataclass
class MeetingSelectionState:
    chat_id: int
    message_id: int
    selections: set[str] = field(default_factory=set)


def build_meeting_keyboard(selections: set[str]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for key, label in MEETING_OPTIONS:
        prefix = "✅ " if key in selections else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{prefix}{label}",
                    callback_data=f"action:meeting:toggle:{key}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="Подтвердить",
                callback_data="action:meeting:confirm",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


SELF_ASSESSMENT_QUESTIONS = [
    "Знание политик и рекомендаций компании",
    "Определение типа ситуации и выбор сценария встречи",
    "Формулирование цели встречи и ожидаемого результата",
    "Формирование повестки и планирование встречи",
    "Отличие фактов от оценочных суждений",
    "Объяснение последствий поведения для команды",
    "Структурирование обратной связи (EECC)",
    "Предположение возражений сотрудника",
    "Выбор фокусных компетенций для развития",
    "Установление контакта в начале встречи",
    "Распознавание базовых эмоций по контексту",
    "Возврат в конструктивное русло при эмоциональных проявлениях",
    "Развивающий диалог и формирование целей развития",
    "Работа с возражениями",
]
SELF_ASSESSMENT_ANSWER_KEYS = {
    "practice": "Нужна практика 💪",
    "confident": "Уверен ✅",
    "unknown": "Не знаком ❓",
}


@dataclass
class SelfAssessmentState:
    chat_id: int
    question_index: int = 0
    answers: list[str] = field(default_factory=list)
    question_message_id: int | None = None


def build_self_assessment_keyboard(question_index: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=SELF_ASSESSMENT_ANSWER_KEYS["practice"],
                    callback_data=f"action:self_assessment:answer:{question_index}:practice",
                )
            ],
            [
                InlineKeyboardButton(
                    text=SELF_ASSESSMENT_ANSWER_KEYS["confident"],
                    callback_data=f"action:self_assessment:answer:{question_index}:confident",
                ),
                InlineKeyboardButton(
                    text=SELF_ASSESSMENT_ANSWER_KEYS["unknown"],
                    callback_data=f"action:self_assessment:answer:{question_index}:unknown",
                ),
            ],
        ]
    )


BLOCK1_QUESTIONS = [
    {
        "text": "Сотрудник показывает высокие результаты 2 года подряд, но на текущей позиции меньше года. Можно ли его продвигать?",
        "answers": [
            {"key": "yes", "text": "Да, можно", "feedback": "✅ Точно! По политике нужен минимум 1 год на позиции перед промо."},
            {"key": "no", "text": "Нет, нужен год на позиции", "feedback": "🤔 Политика действительно требует 1 год, если нет исключения от калибровки."},
            {"key": "calibration", "text": "Нужно решение калибровки", "feedback": "💡 Верно: исключения через калибровку с HR обсуждаются на комиссии."},
        ],
        "hint": "Новый сотрудник получает продвижение только после 1 года на позиции, если нет отдельного решения.",
    },
    {
        "text": "Можно ли обсудить продвижение сотрудника, если он получает высокий рейтинг по результатам, но его карьерный маркер не подтвержден?",
        "answers": [
            {"key": "yes", "text": "Да, можно", "feedback": "✅ Хорошо: в случае подтверждения маркера продвижение возможно."},
            {"key": "no", "text": "Нет, нужен маркер и калибровка", "feedback": "🤔 Правильно: маркер и калибровочный процесс подтверждают, что ресурс готов."},
            {"key": "review", "text": "Нужно решение калибровки", "feedback": "💡 Подтверждение через калибровку устраняет сомнения и фиксирует критерии."},
        ],
        "hint": "Карьерный маркер и калибровка важны для перехода на следующую роль.",
    },
    {
        "text": "Нужно ли знакомить сотрудника с принципами EECC и ННО до самой встречи?",
        "answers": [
            {"key": "yes", "text": "Да, можно", "feedback": "✅ Супер: прозрачность повышает доверие и снижает стресс."},
            {"key": "no", "text": "Нет, не обязательно", "feedback": "🤔 Часто полезно хотя бы кратко озвучить подход для понимания клиента."},
            {"key": "default", "text": "Нужно решение калибровки", "feedback": "💡 Тема оставлена для гибкой трактовки на основе ситуации."},
        ],
        "hint": "Сотрудник должен понимать, как будет строиться обратная связь (тест EECC).",
    },
]


@dataclass
class Block1State:
    chat_id: int
    question_index: int = 0
    question_message_id: int | None = None
    answers: list[tuple[int, str]] = field(default_factory=list)

def build_block1_keyboard(question_index: int) -> InlineKeyboardMarkup:
    question = BLOCK1_QUESTIONS[question_index]
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for answer in question["answers"]:
        row.append(
            InlineKeyboardButton(
                text=answer["text"],
                callback_data=f"action:block1:answer:{question_index}:{answer['key']}",
            )
        )
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append(
        [
            InlineKeyboardButton(
                text="💡 Подсказка",
                callback_data=f"action:block1:hint:{question_index}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_block1_feedback_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Понятно ➡️",
                    callback_data="action:block1:next",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🏠 Главное меню",
                    callback_data="action:post_onboarding_menu",
                ),
                InlineKeyboardButton(
                    text="Пропустить блок",
                    callback_data="action:block1:skip",
                ),
            ],
        ]
    )


BLOCK2_CASE_OVERVIEW = [
    "🎯 Блок 2: Подготовка к встрече",
    "👤 Сотрудник: middle-разработчик, 2 года в компании",
    "📊 Результаты: сроки, качество и отказ от инициатив",
    "🗣 Калибровка: «Оберегать» — ценим как эксперта, делаем ставку на горизонтальное развитие",
]

BLOCK2_SCENARIO_OPTIONS = [
    ("scenario1", "1️⃣ Результаты совпали или превзошли ожидания"),
    ("scenario2", "2️⃣ Недостаточный результат вместо хорошего"),
    ("scenario3", "3️⃣ Хороший результат вместо сверхрезультата"),
]

BLOCK2_SCENARIO_FEEDBACK = {
    "scenario1": "✅ Верно, сценарий 1 — фокус на результатах и развитие",
    "scenario2": "🤔 Это сценарий 2 — нужно работать над ожиданиями сотрудника",
    "scenario3": "💡 Сценарий 3 — хорошо уточнять ценности и цели",
}

BLOCK2_AGENDA_OPTION_DEFS = [
    ("contribution", "Признание вклада"),
    ("review_results", "Обсуждение результатов ревью"),
    ("feedback_zones", "Обратная связь по зонам роста"),
    ("career_expectations", "Обсуждение карьерных ожиданий"),
    ("idp", "План развития (ИПР)"),
    ("salary", "Обсуждение зарплаты"),
    ("next_steps", "Следующие шаги и договорённости"),
]
BLOCK2_AGENDA_LABELS = {key: label for key, label in BLOCK2_AGENDA_OPTION_DEFS}

BLOCK2_FACT_STATEMENTS = [
    "Ты не проявляешь инициативу",
    "В последние 3 месяца ты взял 8 знакомых задач и 0 новых",
    "Ты отказал Ивану в помощи с code review дважды",
    "Ты не хочешь расти",
    "На 1-1 ты сказал: «Не хочу брать ответственность за джуниоров»",
]

BLOCK2_FACT_FEEDBACK = {
    "fact": "✅ Точно, это факт. Подкрепляй конкретными данными.",
    "interpret": "❌ Это оценка — уточни примеры поведения и конкретные наблюдения.",
}

BLOCK2_FOCUS_COMPETENCIES = [
    ("strategic", "Стратегическое мышление"),
    ("team_dev", "Развитие команды"),
    ("influence", "Влияние и коммуникация"),
    ("decisions", "Принятие решений"),
    ("adaptability", "Адаптивность"),
    ("results", "Результативность"),
]
BLOCK2_FOCUS_LABELS = {key: label for key, label in BLOCK2_FOCUS_COMPETENCIES}


@dataclass
class Block2State:
    chat_id: int
    step: str = "intro"
    scenario_choice: str | None = None
    goal_text: str | None = None
    agenda_selections: set[str] = field(default_factory=set)
    agenda_message_id: int | None = None
    fact_index: int = 0
    facts_answers: list[tuple[int, str]] = field(default_factory=list)
    fact_message_id: int | None = None
    consequences_text: str | None = None
    eecc_text: str | None = None
    objections_text: str | None = None
    focus_selections: set[str] = field(default_factory=set)
    focus_message_id: int | None = None


def build_block2_agenda_keyboard(state: Block2State) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for key, label in BLOCK2_AGENDA_OPTION_DEFS:
        prefix = "✅ " if key in state.agenda_selections else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{prefix}{label}",
                    callback_data=f"action:block2:agenda:toggle:{key}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="➡️ Проверить",
                callback_data="action:block2:agenda:check",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_block2_fact_keyboard(index: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Факт ✅",
                    callback_data=f"action:block2:fact:{index}:fact",
                ),
                InlineKeyboardButton(
                    text="Оценка ❌",
                    callback_data=f"action:block2:fact:{index}:interpret",
                ),
            ]
        ]
    )


def build_block2_fact_next_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Понятно ➡️",
                    callback_data="action:block2:next_fact",
                )
            ]
        ]
    )


def build_block2_focus_keyboard(state: Block2State) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for key, label in BLOCK2_FOCUS_COMPETENCIES:
        prefix = "✅ " if key in state.focus_selections else ""
        row.append(
            InlineKeyboardButton(
                text=f"{prefix}{label}",
                callback_data=f"action:block2:focus:toggle:{key}",
            )
        )
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append(
        [
            InlineKeyboardButton(
                text="➡️ Проверить",
                callback_data="action:block2:focus:check",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)

async def send_chunks(
    bot: Bot,
    chat_id: int,
    text: str,
    reply_markup: Any | None,
    max_len: int,
    parse_mode: ParseMode | None = None,
) -> None:
    parts = list(chunk_text(text, max_len))
    for idx, part in enumerate(parts):
        markup = reply_markup if idx == len(parts) - 1 else None
        try:
            await bot.send_message(chat_id, part, reply_markup=markup, parse_mode=parse_mode)
        except TelegramBadRequest as exc:
            logger.warning("failed to send message with HTML parse: %s", exc)
            safe_part = part.replace("<br/>", "\n").replace("<br>", "\n")
            try:
                await bot.send_message(
                    chat_id, safe_part, reply_markup=markup, parse_mode=parse_mode
                )
            except TelegramBadRequest:
                await bot.send_message(chat_id, safe_part, reply_markup=markup, parse_mode=None)


async def main() -> None:
    settings = Settings()
    logging.getLogger().setLevel(settings.log_level)

    bot = Bot(
        settings.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    session = aiohttp.ClientSession()
    backend_client = BackendClient(
        base_url=settings.backend_url,
        token=settings.backend_token,
        session=session,
        timeout_seconds=settings.request_timeout_seconds,
    )
    transcriber = Transcriber(settings=settings, session=session)
    deduplicator = UpdateDeduplicator()
    last_sent_signatures: dict[int, tuple[str, str, str]] = {}
    pending_lock = asyncio.Lock()
    pending_chats: set[int] = set()
    pending_notice_ts: dict[int, float] = {}
    meeting_selection_states: dict[int, MeetingSelectionState] = {}
    self_assessment_states: dict[int, SelfAssessmentState] = {}
    block1_states: dict[int, Block1State] = {}
    block2_states: dict[int, Block2State] = {}
    chat_context: dict[int, dict[str, Any]] = {}

    def get_chat_context(chat_id: int) -> dict[str, Any]:
        default = {
            "modules_unlocked": False,
            "block1_completed": False,
            "block2_completed": False,
            "onboarding_done": False,
            "current_block": None,
            "experience_level": None,
            "selected_scenarios": [],
            "skill_readiness": [],
        }
        return chat_context.setdefault(chat_id, default)
    LEGACY_MENU_TEXT_SNIPPETS = (
        "Принял сообщение",
        "Перейти к тренажёру",
        "Начать диагностику",
        "Напоминания",
        "Сфера деятельности",
    )
    LEGACY_MENU_BUTTONS = {
        "Перейти к тренажеру",
        "Прогресс",
        "В меню",
        "Начать диагностику",
        "Перейти к тренажеру",
        "Напоминания",
        "Сфера деятельности",
    }

    def is_legacy_menu_action(action: dict[str, Any]) -> bool:
        text = str(action.get("text") or "")
        if any(snippet in text for snippet in LEGACY_MENU_TEXT_SNIPPETS):
            return True
        keyboard_raw = None
        if isinstance(action.get("keyboard"), dict):
            keyboard_raw = action["keyboard"].get("inline")
        else:
            keyboard_raw = action.get("keyboard")
        if keyboard_raw:
            for row in keyboard_raw:
                for item in row:
                    label = str(item.get("text") or "").strip()
                    if label in LEGACY_MENU_BUTTONS:
                        return True
        return False

    async def send_post_onboarding_menu(chat_id: int, intro: str | None = None) -> None:
        context = get_chat_context(chat_id)
        text = intro or "Что тренировать дальше?"
        buttons: list[list[InlineKeyboardButton]] = []
        if context.get("block1_completed"):
            buttons.append(
                [
                    InlineKeyboardButton(
                        text="📋 Блок 1: Нормативы ✅",
                        callback_data="action:start:block1",
                    )
                ]
            )
        else:
            buttons.append(
                [
                    InlineKeyboardButton(
                        text="📋 Блок 1: Нормативы",
                        callback_data="action:start:block1",
                    )
                ]
            )
        if context.get("modules_unlocked"):
            block2_label = "🎯 Блок 2: Подготовка к встрече"
            if context.get("block2_completed"):
                block2_label += " ✅"
            buttons.append(
                [
                    InlineKeyboardButton(
                        text=block2_label,
                        callback_data="action:start:block2",
                    ),
                    InlineKeyboardButton(
                        text="💬 Блок 3: Сложные моменты",
                        callback_data="action:start:block3",
                    ),
                ]
            )
        else:
            buttons.append(
                [
                    InlineKeyboardButton(
                        text="🎯 Блок 2: Подготовка к встрече (скоро)",
                        callback_data="action:start:block2",
                    )
                ]
            )
        buttons.append(
            [
                InlineKeyboardButton(
                    text="📊 Мой прогресс",
                    callback_data="action:navigation:progress",
                )
            ]
        )
        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        await bot.send_message(chat_id, text, reply_markup=keyboard)

    async def send_block1_question(state: Block1State) -> None:
        question = BLOCK1_QUESTIONS[state.question_index]
        text = (
            "📋 Блок 1 — Нормативы\n"
            f"({state.question_index + 1}/{len(BLOCK1_QUESTIONS)}) {question['text']}"
        )
        message = await bot.send_message(
            state.chat_id,
            text,
            reply_markup=build_block1_keyboard(state.question_index),
        )
        state.question_message_id = message.message_id

    async def begin_block1(chat_id: int) -> None:
        context = get_chat_context(chat_id)
        context["current_block"] = 1
        context["block1_completed"] = False
        state = Block1State(chat_id=chat_id)
        block1_states[chat_id] = state
        await bot.send_message(
            chat_id,
            "📋 Блок 1: Нормативы компании\nУбедимся, что ты знаком с ключевыми правилами (3-5 минут).",
        )
        await send_block1_question(state)

    async def finish_block1(chat_id: int) -> None:
        context = get_chat_context(chat_id)
        context["modules_unlocked"] = True
        context["block1_completed"] = True
        context["current_block"] = None
        block1_states.pop(chat_id, None)
        await bot.send_message(
            chat_id,
            "🎉 Отлично, Блок 1 завершён! Навыки нормативов зафиксированы.",
        )
        await send_post_onboarding_menu(
            chat_id, "Что хочешь потренировать дальше?"
        )

    async def send_action_event(
        user_id: int,
        username: str | None,
        chat_id: int,
        action: str,
        raw: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        payload = build_event_payload(
            update_id=int(raw.get("request_id") or 0),
            from_user=type("U", (), {"id": user_id, "username": username}),
            chat_id=chat_id,
            event={"type": "action", "action": action},
            raw=raw,
            client_ts=datetime.now(timezone.utc),
            meta=meta,
        )
        return await send_to_backend(payload)

    async def send_self_assessment_question(state: SelfAssessmentState) -> None:
        if state.question_index >= len(SELF_ASSESSMENT_QUESTIONS):
            return
        question_text = SELF_ASSESSMENT_QUESTIONS[state.question_index]
        formatted_text = f"({state.question_index + 1}) {question_text}"
        keyboard = build_self_assessment_keyboard(state.question_index)
        if state.question_message_id:
            try:
                await bot.edit_message_text(
                    chat_id=state.chat_id,
                    message_id=state.question_message_id,
                    text=formatted_text,
                    reply_markup=keyboard,
                )
                return
            except TelegramBadRequest:
                state.question_message_id = None
        message = await bot.send_message(
            state.chat_id,
            formatted_text,
            reply_markup=keyboard,
        )
        state.question_message_id = message.message_id

    async def begin_self_assessment(chat_id: int) -> None:
        if not SELF_ASSESSMENT_QUESTIONS:
            return
        context = get_chat_context(chat_id)
        context["onboarding_done"] = True
        await bot.send_message(chat_id, "Самооценка по навыкам")
        state = SelfAssessmentState(chat_id=chat_id)
        self_assessment_states[chat_id] = state
        await send_self_assessment_question(state)


    async def begin_block2(chat_id: int) -> None:
        context = get_chat_context(chat_id)
        if not context.get('block1_completed'):
            await bot.send_message(chat_id, 'Сначала пройди Блок 1: нормативы.')
            return
        if context.get('current_block') == 2:
            await bot.send_message(chat_id, 'Ты уже в процессе блока 2.')
            return
        context['current_block'] = 2
        state = Block2State(chat_id=chat_id)
        block2_states[chat_id] = state
        for line in BLOCK2_CASE_OVERVIEW:
            await bot.send_message(chat_id, line)
        await bot.send_message(
            chat_id,
            '🎯 Готов начать подготовку? Нажми «Начать подготовку 🎯»',
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text='Начать подготовку 🎯',
                            callback_data='action:block2:start',
                        )
                    ]
                ]
            ),
        )

    async def send_block2_scenario(state: Block2State) -> None:
        state.step = 'scenario'
        rows = [
            [
                InlineKeyboardButton(
                    text=text,
                    callback_data=f'action:block2:scenario:{key}',
                )
            ]
            for key, text in BLOCK2_SCENARIO_OPTIONS
        ]
        rows.append(
            [
                InlineKeyboardButton(
                    text='💡 Подсказка',
                    callback_data='action:block2:scenario:hint',
                )
            ]
        )
        await bot.send_message(
            state.chat_id,
            'К какому типу ситуации относится этот кейс?',
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        )

    async def send_block2_goal_prompt(state: Block2State) -> None:
        state.step = 'goal'
        await bot.send_message(
            state.chat_id,
            'Сформулируй цель встречи с этим сотрудником. Напиши текст или голосом.',
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text='Пример цели 💡',
                            callback_data='action:block2:goal:example',
                        )
                    ]
                ]
            ),
        )

    async def send_block2_agenda_prompt(state: Block2State) -> None:
        state.step = 'agenda'
        message = await bot.send_message(
            state.chat_id,
            'Что важно включить в повестку? Выбери нужные пункты 👇',
            reply_markup=build_block2_agenda_keyboard(state),
        )
        state.agenda_message_id = message.message_id

    async def send_block2_fact_statement(state: Block2State) -> None:
        if state.fact_index >= len(BLOCK2_FACT_STATEMENTS):
            await send_block2_consequences_prompt(state)
            return
        state.step = 'fact'
        statement = BLOCK2_FACT_STATEMENTS[state.fact_index].replace("\\n", "\n")
        message = await bot.send_message(
            state.chat_id,
            f"Факт или оценка?\n{statement}",
            reply_markup=build_block2_fact_keyboard(state.fact_index),
        )
        state.fact_message_id = message.message_id if hasattr(state, 'fact_message_id') else None

    async def send_block2_consequences_prompt(state: Block2State) -> None:
        state.step = 'consequences'
        await bot.send_message(
            state.chat_id,
            'Как объяснишь влияние отказа от менторства на команду и компанию? Напиши пару предложений.',
        )

    async def send_block2_eecc_prompt(state: Block2State) -> None:
        state.step = 'eecc'
        await bot.send_message(
            state.chat_id,
            'Структурируй обратную связь по модели EECC: Example → Effect → Change → Continue.',
        )

    async def send_block2_objections_prompt(state: Block2State) -> None:
        state.step = 'objections'
        await bot.send_message(
            state.chat_id,
            'Какие возражения может выдвинуть сотрудник? Напиши 2-3 варианта и как ответишь.',
        )

    async def send_block2_focus_prompt(state: Block2State) -> None:
        state.step = 'focus'
        message = await bot.send_message(
            state.chat_id,
            'Выбери 2-3 фокусных компетенции для развития.',
            reply_markup=build_block2_focus_keyboard(state),
        )
        state.focus_message_id = message.message_id

    async def finish_block2(chat_id: int) -> None:
        context = get_chat_context(chat_id)
        context['current_block'] = None
        context['block2_completed'] = True
        block2_states.pop(chat_id, None)
        await bot.send_message(
            chat_id,
            '✅ Отлично! Блок 2 завершён. Ты отработал подготовку к встречам.',
        )
        await send_post_onboarding_menu(chat_id, 'Что тренировать дальше?')

    async def handle_block2_text_input(message: Message, state: Block2State) -> None:
        reply = (message.text or '').strip()
        if not reply:
            return
        if state.step == 'goal':
            state.goal_text = reply
            await bot.send_message(message.chat.id, '💬 Принял цель. Теперь выбери элементы повестки.')
            await send_block2_agenda_prompt(state)
        elif state.step == 'consequences':
            state.consequences_text = reply
            await bot.send_message(message.chat.id, '💬 Спасибо! Теперь структурируй EECC.')
            await send_block2_eecc_prompt(state)
        elif state.step == 'eecc':
            state.eecc_text = reply
            await bot.send_message(message.chat.id, '💬 Принял формат. Предположи возражения.')
            await send_block2_objections_prompt(state)
        elif state.step == 'objections':
            state.objections_text = reply
            await bot.send_message(message.chat.id, '💬 Спасибо! Осталось выбрать компетенции.')
            await send_block2_focus_prompt(state)

    async def try_set_pending(chat_id: int) -> bool:
        async with pending_lock:
            if chat_id in pending_chats:
                return False
            pending_chats.add(chat_id)
            return True

    async def clear_pending(chat_id: int) -> None:
        async with pending_lock:
            pending_chats.discard(chat_id)
            pending_notice_ts.pop(chat_id, None)

    async def should_warn_pending(chat_id: int, cooldown: float = 5.0) -> bool:
        now = time.monotonic()
        async with pending_lock:
            last = pending_notice_ts.get(chat_id, 0)
            if now - last < cooldown:
                return False
            pending_notice_ts[chat_id] = now
            return True

    async def send_to_backend(payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return await backend_client.send_event(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to reach backend: %s", exc)
            return None

    async def send_actions(
        backend_response: dict[str, Any] | None, message_to_edit: Message | None = None
    ) -> None:
        if not backend_response:
            return
        actions = backend_response.get("actions") or []
        legacy_chat_ids: set[int] = set()
        filtered_actions: list[dict[str, Any]] = []
        for action in actions:
            if is_legacy_menu_action(action):
                chat_id = action.get("chat_id")
                if isinstance(chat_id, int):
                    legacy_chat_ids.add(chat_id)
                continue
            filtered_actions.append(action)
        actions = filtered_actions
        best_by_text: dict[tuple[str, str], dict[str, Any]] = {}
        for action in actions:
            if action.get("type") != "send_message":
                continue
            chat_id = action.get("chat_id")
            if chat_id is None:
                continue
            text = str(action.get("text") or "").strip()
            if not text:
                continue
            parse_mode_value = str(action.get("parse_mode") or ParseMode.HTML)
            keyboard_raw = None
            if isinstance(action.get("keyboard"), dict):
                keyboard_raw = action["keyboard"].get("inline")
            elif action.get("keyboard"):
                keyboard_raw = action.get("keyboard")
            key = (text, parse_mode_value)
            existing = best_by_text.get(key)
            has_keyboard = bool(keyboard_raw)
            if existing is None or (has_keyboard and not existing.get("__has_keyboard")):
                best = dict(action)
                best["__keyboard_raw"] = keyboard_raw
                best["__has_keyboard"] = has_keyboard
                best_by_text[key] = best
        for action in best_by_text.values():
            keyboard_raw = action.pop("__keyboard_raw", None)
            keyboard = build_keyboard(keyboard_raw)
            parse_mode_value = action.get("parse_mode") or ParseMode.HTML
            try:
                parse_mode = ParseMode(parse_mode_value)
            except Exception:  # noqa: BLE001
                parse_mode = ParseMode.HTML
            signature = (
                str(action.get("text") or ""),
                json.dumps(keyboard_raw, sort_keys=True) if keyboard_raw else "",
                str(parse_mode),
            )
            if not message_to_edit:
                last_sig = last_sent_signatures.get(action["chat_id"])
                if last_sig == signature:
                    continue
            if message_to_edit and message_to_edit.chat and message_to_edit.message_id:
                try:
                    await bot.edit_message_text(
                        text=action["text"],
                        chat_id=message_to_edit.chat.id,
                        message_id=message_to_edit.message_id,
                        reply_markup=keyboard,
                        parse_mode=parse_mode,
                    )
                    message_to_edit = None
                    last_sent_signatures[action["chat_id"]] = signature
                    continue
                except TelegramBadRequest:
                    message_to_edit = None

            await send_chunks(
                bot,
                action["chat_id"],
                action["text"],
                keyboard,
                settings.max_tg_message_length,
                parse_mode=parse_mode,
            )
            last_sent_signatures[action["chat_id"]] = signature
        for legacy_chat_id in legacy_chat_ids:
            context = get_chat_context(legacy_chat_id)
            if context.get("current_block"):
                continue
            await send_post_onboarding_menu(
                legacy_chat_id, "Что хочешь потренировать дальше?"
            )

    async def answer_backend(
        chat_id: int,
        backend_response: dict[str, Any] | None,
        message_to_edit: Message | None = None,
    ) -> None:
        if not backend_response:
            error_text = "Техническая ошибка при обращении к бэкенду. Попробуйте позже."
            if message_to_edit and message_to_edit.chat and message_to_edit.message_id:
                try:
                    await bot.edit_message_text(
                        text=error_text,
                        chat_id=message_to_edit.chat.id,
                        message_id=message_to_edit.message_id,
                    )
                    return
                except TelegramBadRequest:
                    message_to_edit = None
            await bot.send_message(chat_id, error_text)
            return
        if "actions" in backend_response:
            await send_actions(backend_response, message_to_edit=message_to_edit)
            return

        # backward-compatible branch
        text = str(backend_response.get("text") or "").strip()
        if not text:
            text = "Ответ пустой. Попробуйте ещё раз."
        keyboard = build_keyboard(backend_response.get("keyboard"))
        if is_legacy_menu_action({"text": text, "keyboard": backend_response.get("keyboard")}):
            await send_post_onboarding_menu(chat_id, "Что хочешь потренировать дальше?")
            return
        if message_to_edit and message_to_edit.chat and message_to_edit.message_id:
            try:
                await bot.edit_message_text(
                    text=text,
                    chat_id=message_to_edit.chat.id,
                    message_id=message_to_edit.message_id,
                    reply_markup=keyboard,
                    parse_mode=ParseMode.HTML,
                )
                return
            except TelegramBadRequest:
                message_to_edit = None
        await send_chunks(bot, chat_id, text, keyboard, settings.max_tg_message_length, parse_mode=ParseMode.HTML)

    @dp.message(CommandStart())
    async def handle_start(message: Message) -> None:
        intro_messages = [
            "Привет! 👋 Я помогу тебе подготовиться к встречам с сотрудниками после performance и talent review 💼",
            "Мы разберём сложные сценарии, потренируем фразы и структуру встреч, чтобы ты чувствовал себя увереннее 💪",
        ]
        for text in intro_messages:
            await bot.send_message(message.chat.id, text)

        await bot.send_message(
            message.chat.id,
            "Сколько у тебя опыта руководства?",
            reply_markup=build_experience_keyboard(),
        )

        payload = build_event_payload(
            update_id=message.message_id,
            from_user=message.from_user,
            chat_id=message.chat.id,
            event={
                "type": "action",
                "action": "start",
                "text": settings.default_reply_text,
            },
            raw=minimal_raw_message(message),
            client_ts=message.date,
        )

        async def fire_start_event() -> None:
            try:
                await send_to_backend(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("start event failed: %s", exc)

        asyncio.create_task(fire_start_event())

    @dp.message(F.text == "/menu")
    @dp.message(F.text == "/help")
    async def handle_menu(message: Message) -> None:
        context = get_chat_context(message.chat.id)
        if context.get("onboarding_done"):
            await send_post_onboarding_menu(message.chat.id)
            return
        await bot.send_message(
            message.chat.id,
            "Выберите действие:",
            reply_markup=main_menu_keyboard(),
        )

    @dp.message(F.text == "/diagnostic")
    async def handle_diagnostic_command(message: Message) -> None:
        backend_resp = await send_action_event(
            user_id=message.from_user.id,
            username=message.from_user.username,
            chat_id=message.chat.id,
            action="diagnostic:start",
            raw={
                "request_id": message.message_id,
                "command": "/diagnostic",
            },
        )
        if backend_resp:
            await answer_backend(message.chat.id, backend_resp)
        else:
            await bot.send_message(
                message.chat.id,
                "Запускаю диагностику...",
                reply_markup=main_menu_keyboard(),
            )

    @dp.message(F.voice)
    async def handle_voice(message: Message) -> None:
        if not message.voice:
            return
        if not await try_set_pending(message.chat.id):
            if await should_warn_pending(message.chat.id):
                await bot.send_message(
                    message.chat.id,
                    "Уже готовлю ответ на предыдущий запрос. Подожди пару секунд.",
                )
            return
        try:
            start_ts = time.monotonic()
            voice_file = await transcriber.download_voice(bot, message.voice)
            transcript: str | None = "[voice message]"
            confidence: float | None = None
            try:
                if transcriber.settings.deepgram_api_key or transcriber.settings.transcribe_url:
                    result = await transcriber.transcribe(voice_file)
                    transcript = result.text
                    confidence = result.confidence
                    logger.info(
                        "voice transcription done chat=%s len=%s confidence=%s",
                        message.chat.id,
                        len(transcript or ""),
                        confidence,
                    )
                else:
                    logger.info("ASR is not set, sending placeholder text")
                    transcript = "[voice message]"
            except Exception as exc:  # noqa: BLE001
                logger.warning("transcription failed: %s", exc)
                await bot.send_message(
                    message.chat.id,
                    "Не удалось распознать голос. Повторите голосом или отправьте текст.",
                )
                return
            finally:
                voice_file.unlink(missing_ok=True)

            duration_ms = int((time.monotonic() - start_ts) * 1000)
            payload = build_event_payload(
                update_id=message.message_id,
                from_user=message.from_user,
                chat_id=message.chat.id,
                event={
                    "type": "text",
                    "text": transcript,
                    "source": "voice",
                },
                raw=minimal_raw_message(message),
                client_ts=message.date,
                meta={
                    "asr": {"confidence": confidence},
                    "voice_seconds": message.voice.duration,
                    "asr_duration_ms": duration_ms,
                },
            )
            try:
                backend_resp = await send_to_backend(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("backend request failed: %s", exc)
                backend_resp = None
            await answer_backend(message.chat.id, backend_resp)
        finally:
            await clear_pending(message.chat.id)

    @dp.message(F.text)
    async def handle_text(message: Message) -> None:
        block2_state = block2_states.get(message.chat.id)
        if block2_state and block2_state.step in {"goal", "consequences", "eecc", "objections"}:
            await handle_block2_text_input(message, block2_state)
            return
        if message.text and message.text.startswith("/"):
            return
        if not await try_set_pending(message.chat.id):
            if await should_warn_pending(message.chat.id):
                await bot.send_message(
                    message.chat.id,
                    "Уже готовлю ответ на предыдущий запрос. Подожди пару секунд.",
                )
            return
        try:
            loading_message: Message | None = None
            try:
                loading_message = await bot.send_message(
                    message.chat.id,
                    random.choice(LOADING_MESSAGES),
                )
            except Exception:
                loading_message = None
            payload = build_event_payload(
                update_id=message.message_id,
                from_user=message.from_user,
                chat_id=message.chat.id,
                event={
                    "type": "text",
                    "text": message.text,
                },
                raw=minimal_raw_message(message),
                client_ts=message.date,
            )
            try:
                backend_resp = await send_to_backend(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("backend request failed: %s", exc)
                backend_resp = None
            await answer_backend(message.chat.id, backend_resp, message_to_edit=loading_message)
        finally:
            await clear_pending(message.chat.id)

    @dp.callback_query()
    async def handle_callback(callback: CallbackQuery) -> None:
        data = callback.data or ""
        # block callbacks while pending to avoid double-processing
        if callback.message and callback.message.chat and callback.message.chat.id in pending_chats:
            if await should_warn_pending(callback.message.chat.id):
                await callback.answer("Готовлю ответ на предыдущий запрос, подожди", show_alert=False)
            return
        loading_message: Message | None = None
        action_name: str | None = None
        chat_id = callback.message.chat.id if callback.message else 0
        if data.startswith("action:"):
            action_name = data.split("action:", 1)[1]
        else:
            action_name = None
        if action_name == "post_onboarding_menu":
            await send_post_onboarding_menu(chat_id)
            await callback.answer()
            return
        if action_name == "start:block1":
            await begin_block1(chat_id)
            await callback.answer()
            return
        if action_name == "start:block2":
            await begin_block2(chat_id)
            await callback.answer()
            return
        if action_name == "start:block3":
            await bot.send_message(chat_id, "Блок 3 пока недоступен.")
            await callback.answer()
            return
        if action_name == "navigation:progress":
            context = get_chat_context(chat_id)
            progress_lines = [
                "📊 Прогресс",
                f"Блок 1: {'✅' if context.get('block1_completed') else '⏳'}",
                f"Блок 2: {'✅' if context.get('block2_completed') else '⏳'}",
                f"Модули разблокированы: {'Да' if context.get('modules_unlocked') else 'Нет'}",
            ]
            await bot.send_message(chat_id, "\n".join(progress_lines))
            await callback.answer()
            return
        if action_name == "block2:start":
            state = block2_states.get(chat_id)
            if not state:
                await callback.answer("Сначала начни Блок 2 через меню.", show_alert=True)
                return
            await send_block2_scenario(state)
            await callback.answer()
            return
        if action_name and action_name.startswith("block2:scenario:"):
            parts = action_name.split(":")
            if len(parts) != 3:
                await callback.answer()
                return
            choice = parts[2]
            state = block2_states.get(chat_id)
            if not state:
                await callback.answer()
                return
            if choice == "hint":
                await callback.answer("Выбирай сценарий по ожиданиям сотрудника и результатам калибровки.", show_alert=True)
                return
            if choice not in {key for key, _ in BLOCK2_SCENARIO_OPTIONS}:
                await callback.answer()
                return
            state.scenario_choice = choice
            await bot.send_message(chat_id, BLOCK2_SCENARIO_FEEDBACK.get(choice, "Отлично!"))
            await send_block2_goal_prompt(state)
            await callback.answer()
            return
        if action_name == "block2:goal:example":
            await bot.send_message(chat_id, 'Пример цели: "Подтвердить ценность вклада, обсудить варианты горизонтального развития и согласовать два шага на квартал".')
            await callback.answer()
            return
        if action_name and action_name.startswith("block2:agenda:toggle:"):
            parts = action_name.split(":")
            if len(parts) != 4:
                await callback.answer()
                return
            option_key = parts[3]
            state = block2_states.get(chat_id)
            if not state or state.step != "agenda":
                await callback.answer()
                return
            option_label = BLOCK2_AGENDA_LABELS.get(option_key)
            if option_label is None:
                await callback.answer()
                return
            if option_key in state.agenda_selections:
                state.agenda_selections.remove(option_key)
            else:
                state.agenda_selections.add(option_key)
            if state.agenda_message_id:
                try:
                    await bot.edit_message_reply_markup(
                        chat_id=chat_id,
                        message_id=state.agenda_message_id,
                        reply_markup=build_block2_agenda_keyboard(state),
                        )
                except TelegramBadRequest:
                    pass
            await callback.answer()
            return
        if action_name == "block2:agenda:check":
            state = block2_states.get(chat_id)
            if not state or not state.agenda_selections:
                await callback.answer("Выбери хотя бы один пункт.", show_alert=True)
                return
            await bot.send_message(
                chat_id,
                "Отлично, ты выбрал: "
                + ", ".join(
                    BLOCK2_AGENDA_LABELS.get(key, key) for key in state.agenda_selections
                ),
            )
            await send_block2_fact_statement(state)
            await callback.answer()
            return
        if action_name and action_name.startswith("block2:fact:"):
            parts = action_name.split(":")
            if len(parts) != 4:
                await callback.answer()
                return
            try:
                fact_index = int(parts[2])
            except ValueError:
                await callback.answer()
                return
            choice = parts[3]
            if choice not in {"fact", "interpret"}:
                await callback.answer()
                return
            state = block2_states.get(chat_id)
            if not state or state.fact_index != fact_index:
                await callback.answer()
                return
            state.facts_answers.append((fact_index, choice))
            await bot.send_message(
                chat_id,
                BLOCK2_FACT_FEEDBACK[choice],
                reply_markup=build_block2_fact_next_keyboard(),
            )
            await callback.answer()
            return
        if action_name == "block2:next_fact":
            state = block2_states.get(chat_id)
            if not state:
                await callback.answer()
                return
            state.fact_index += 1
            await send_block2_fact_statement(state)
            await callback.answer()
            return
        if action_name and action_name.startswith("block2:focus:toggle:"):
            parts = action_name.split(":")
            if len(parts) != 4:
                await callback.answer()
                return
            option_key = parts[3]
            state = block2_states.get(chat_id)
            if not state or state.step != "focus":
                await callback.answer()
                return
            option_label = BLOCK2_FOCUS_LABELS.get(option_key)
            if option_label is None:
                await callback.answer()
                return
            if option_key in state.focus_selections:
                state.focus_selections.remove(option_key)
            else:
                state.focus_selections.add(option_key)
            if state.focus_message_id:
                try:
                    await bot.edit_message_reply_markup(
                        chat_id=chat_id,
                        message_id=state.focus_message_id,
                        reply_markup=build_block2_focus_keyboard(state),
                        )
                except TelegramBadRequest:
                    pass
            await callback.answer()
            return
        if action_name == "block2:focus:check":
            state = block2_states.get(chat_id)
            if not state or not state.focus_selections:
                await callback.answer("Выбери хотя бы одну компетенцию.", show_alert=True)
                return
            await bot.send_message(
                chat_id,
                "Сфокусирован на: "
                + ", ".join(
                    BLOCK2_FOCUS_LABELS.get(key, key) for key in state.focus_selections
                ),
            )
            await finish_block2(chat_id)
            await callback.answer()
            return
        if action_name and action_name.startswith("block1:hint:"):
            parts = action_name.split(":")
            if len(parts) == 3:
                try:
                    question_index = int(parts[2])
                except ValueError:
                    question_index = None
                if question_index is not None and 0 <= question_index < len(BLOCK1_QUESTIONS):
                    hint = BLOCK1_QUESTIONS[question_index].get("hint")
                    if hint:
                        await callback.answer(hint, show_alert=True)
                        return
            await callback.answer()
            return
        if action_name and action_name.startswith("block1:answer:"):
            parts = action_name.split(":")
            if len(parts) != 4:
                await callback.answer()
                return
            try:
                question_index = int(parts[2])
            except ValueError:
                await callback.answer()
                return
            choice_key = parts[3]
            state = block1_states.get(chat_id)
            if not state or state.question_index != question_index:
                await callback.answer()
                return
            question = BLOCK1_QUESTIONS[question_index]
            answer = next((a for a in question["answers"] if a["key"] == choice_key), None)
            if not answer:
                await callback.answer()
                return
            if state.question_message_id:
                try:
                    await bot.edit_message_reply_markup(
                        chat_id=chat_id,
                        message_id=state.question_message_id,
                        reply_markup=None,
                    )
                except TelegramBadRequest:
                    pass
            state.answers.append((question_index, choice_key))
            await bot.send_message(
                chat_id,
                answer["feedback"],
                reply_markup=build_block1_feedback_keyboard(),
            )
            await callback.answer()
            return
        if action_name == "block1:next":
            state = block1_states.get(chat_id)
            if not state:
                await callback.answer()
                return
            state.question_index += 1
            if state.question_index >= len(BLOCK1_QUESTIONS):
                await finish_block1(chat_id)
                await callback.answer()
                return
            await send_block1_question(state)
            await callback.answer()
            return
        if action_name == "block1:skip":
            block1_states.pop(chat_id, None)
            await send_post_onboarding_menu(chat_id, "Блок 1 пропущен. Что дальше?")
            await callback.answer()
            return
        if action_name and action_name.startswith("experience:"):
            experience_key = action_name.split("experience:", 1)[1]
            context = get_chat_context(chat_id)
            context["experience_level"] = EXPERIENCE_LEVEL_LABELS.get(
                experience_key, experience_key
            )
            context["selected_scenarios"] = []
            if callback.message:
                try:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=callback.message.message_id,
                        text=MEETING_PROMPT,
                        reply_markup=build_meeting_keyboard(set()),
                    )
                except TelegramBadRequest:
                    pass
            meeting_selection_states[chat_id] = MeetingSelectionState(
                chat_id=chat_id,
                message_id=callback.message.message_id if callback.message else 0,
            )

            async def fire_experience_event() -> None:
                try:
                    await send_action_event(
                        user_id=callback.from_user.id,
                        username=callback.from_user.username,
                        chat_id=chat_id,
                        action=action_name,
                        raw={"id": callback.id, "data": data},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("failed to report experience action: %s", exc)

            asyncio.create_task(fire_experience_event())
            await callback.answer()
            return
        if action_name and action_name.startswith("self_assessment:answer:"):
            chat_id = callback.message.chat.id if callback.message else 0
            parts = action_name.split(":")
            if len(parts) != 4:
                await callback.answer()
                return
            try:
                question_index = int(parts[2])
            except ValueError:
                await callback.answer()
                return
            answer_key = parts[3]
            if answer_key not in SELF_ASSESSMENT_ANSWER_KEYS:
                await callback.answer()
                return
            state = self_assessment_states.get(chat_id)
            if not state or state.question_index != question_index:
                await callback.answer()
                return
            if state.question_message_id:
                try:
                    await bot.edit_message_reply_markup(
                        chat_id=chat_id,
                        message_id=state.question_message_id,
                        reply_markup=None,
                    )
                except TelegramBadRequest:
                    pass
            state.answers.append(answer_key)
            state.question_index += 1
            if state.question_index >= len(SELF_ASSESSMENT_QUESTIONS):
                self_assessment_states.pop(chat_id, None)
                context = get_chat_context(chat_id)
                readiness = []
                for idx, key in enumerate(state.answers):
                    question = SELF_ASSESSMENT_QUESTIONS[idx]
                    readiness.append(
                        {
                            "question": question,
                            "answer_key": key,
                            "answer_label": SELF_ASSESSMENT_ANSWER_KEYS[key],
                        }
                    )
                context["skill_readiness"] = readiness
                await bot.send_message(chat_id, "Спасибо! Я записал ответы и продолжу.")
                await begin_block1(chat_id)
                await callback.answer()
                return
            await send_self_assessment_question(state)
            await callback.answer()
            return
        if action_name and action_name.startswith("meeting:toggle:"):
            chat_id = callback.message.chat.id if callback.message else 0
            if chat_id:
                state = meeting_selection_states.get(chat_id)
                if not state:
                    state = MeetingSelectionState(
                        chat_id=chat_id,
                        message_id=callback.message.message_id if callback.message else 0,
                    )
                option_key = action_name.split("meeting:toggle:", 1)[1]
                if option_key not in MEETING_OPTION_KEYS:
                    await callback.answer()
                    return
                if option_key in state.selections:
                    state.selections.remove(option_key)
                else:
                    state.selections.add(option_key)
                meeting_selection_states[chat_id] = state
                if callback.message:
                    try:
                        await bot.edit_message_reply_markup(
                            chat_id=chat_id,
                            message_id=state.message_id,
                            reply_markup=build_meeting_keyboard(state.selections),
                        )
                    except TelegramBadRequest:
                        pass
            await callback.answer()
            return
        if action_name == "meeting:confirm":
            chat_id = callback.message.chat.id if callback.message else 0
            state = meeting_selection_states.get(chat_id)
            if not state or not state.selections:
                await callback.answer("Выбери хотя бы одну тему", show_alert=True)
                return
            meeting_selection_states.pop(chat_id, None)
            context = get_chat_context(chat_id)
            context["selected_scenarios"] = sorted(state.selections)
            backend_resp = await send_action_event(
                user_id=callback.from_user.id,
                username=callback.from_user.username,
                chat_id=chat_id,
                action=action_name,
                raw={
                    "id": callback.id,
                    "data": data,
                    "selections": sorted(state.selections),
                },
                meta={"meeting_choices": sorted(state.selections)},
            )
            await callback.answer()
            if callback.message:
                await answer_backend(
                    callback.message.chat.id,
                    backend_resp,
                    message_to_edit=callback.message,
                )
            await begin_self_assessment(chat_id)
            return
        requires_loading = data.startswith("diag:") or action_name in {
            "diagnostic:start",
            "training:start",
        }
        if requires_loading and callback.message:
            try:
                loading_message = await bot.send_message(
                    callback.message.chat.id,
                    random.choice(LOADING_MESSAGES),
                )
            except Exception:
                loading_message = None
        payload = build_event_payload(
            update_id=callback.message.message_id if callback.message else 0,
            from_user=callback.from_user,
            chat_id=callback.message.chat.id if callback.message else 0,
            event={
                "type": "callback",
                "data": data,
            },
            raw={"id": callback.id, "data": data},
            client_ts=datetime.now(timezone.utc),
        )
        backend_resp: dict[str, Any] | None
        if data.startswith("action:"):
            action_name = data.split("action:", 1)[1]
            backend_resp = await send_action_event(
                user_id=callback.from_user.id,
                username=callback.from_user.username,
                chat_id=callback.message.chat.id if callback.message else 0,
                action=action_name,
                raw={"id": callback.id, "data": data},
            )
        else:
            backend_resp = await send_to_backend(payload)
        await callback.answer()
        if callback.message:
            await answer_backend(callback.message.chat.id, backend_resp, message_to_edit=loading_message or callback.message)

    def build_event_payload(
        update_id: int,
        from_user: Any,
        chat_id: int,
        event: dict[str, Any],
        raw: dict[str, Any],
        client_ts: datetime,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_id = str(uuid.uuid4())
        payload = {
            "event_id": event_id,
            "telegram_update_id": update_id,
            "user": {
                "user_id": from_user.id if from_user else None,
                "chat_id": chat_id,
                "username": from_user.username if from_user else None,
            },
            "event": event,
            "meta": {
                "source": "telegram",
                "client_ts": client_ts.isoformat(),
            },
        }
        if meta:
            payload["meta"].update(meta)
        payload["event"]["raw"] = raw
        return payload

    async def healthcheck(_: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def webhook_handler(request: web.Request) -> web.Response:
        if settings.webhook_secret:
            header_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if header_token != settings.webhook_secret:
                return web.Response(status=401, text="invalid secret")

        raw_data = await request.text()
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return web.Response(status=400, text="bad json")

        try:
            update = Update.model_validate(data)
        except Exception:  # noqa: BLE001
            return web.Response(status=400, text="invalid update")

        if await deduplicator.is_duplicate(update.update_id):
            logger.info("duplicate update %s dropped", update.update_id)
            return web.Response(text="duplicate")

        async def process() -> None:
            try:
                await dp.feed_update(bot, update)
            except Exception as exc:  # noqa: BLE001
                logger.exception("failed to process update %s: %s", update.update_id, exc)

        asyncio.create_task(process())
        return web.Response(text="ok")

    async def push_handler(request: web.Request) -> web.Response:
        if settings.push_token:
            auth_header = request.headers.get("Authorization", "")
            if auth_header != f"Bearer {settings.push_token}":
                return web.Response(status=401, text="unauthorized")
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400, text="bad json")

        try:
            await send_actions(body)
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to process push: %s", exc)
            return web.Response(status=500, text="failed")

        return web.Response(text="ok")

    async def set_bot_commands() -> None:
        commands = [
            BotCommand(command="start", description="Приветствие"),
            BotCommand(command="menu", description="Меню"),
            BotCommand(command="diagnostic", description="Начать диагностику"),
            BotCommand(command="help", description="Подсказка по меню"),
        ]
        try:
            await bot.set_my_commands(commands)
            logger.info("bot commands updated")
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to set bot commands: %s", exc)

    try:
        if settings.use_webhook:
            app = web.Application()
            app.router.add_get("/health", healthcheck)
            app.router.add_post(settings.webhook_path, webhook_handler)
            app.router.add_post("/push", push_handler)

            async def on_startup(_: web.Application) -> None:
                await set_bot_commands()
                await bot.set_webhook(
                    url=settings.webhook_url,
                    secret_token=settings.webhook_secret,
                    drop_pending_updates=True,
                )
                logger.info("webhook set to %s", settings.webhook_url)

            async def on_shutdown(_: web.Application) -> None:
                await bot.session.close()
                await session.close()

            app.on_startup.append(on_startup)
            app.on_shutdown.append(on_shutdown)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, settings.listen_host, settings.listen_port)
            logger.info(
                "starting webhook listener on %s:%s path=%s",
                settings.listen_host,
                settings.listen_port,
                settings.webhook_path,
            )
            await site.start()

            try:
                while True:
                    await asyncio.sleep(3600)
            finally:
                await runner.cleanup()
        else:
            logger.info("starting long polling mode (webhook disabled)")
            try:
                await bot.delete_webhook(drop_pending_updates=True)
            except Exception:  # noqa: BLE001
                logger.debug("failed to delete webhook, continuing with polling")
            await set_bot_commands()
            await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass
        try:
            await session.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
