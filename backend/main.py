from __future__ import annotations

import asyncio
import logging
import sys
import uuid
import json
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ai import evaluate_answer, interpret_diagnostic
from ai_client import AIClient
from config import Settings
from data import (
    DIAGNOSTIC_QUESTIONS,
    TRAINING_CASES_FEEDBACK,
    TRAINING_CASES_IDP,
    SPHERES,
    get_case,
)
from state import StateStore
from database import ConversationDB, SkillTrainingDB, init_db
from skills_data import SKILL_BLOCKS, get_all_blocks, get_block, get_skill, get_all_skills_flat


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Training Bot Backend")
state_store = StateStore()
http_client = httpx.AsyncClient(timeout=15)
ingest_counter = 0
ai_client: Optional[AIClient] = None


class UserModel(BaseModel):
    user_id: int
    chat_id: int
    username: Optional[str] = None


class EventModel(BaseModel):
    type: str
    text: Optional[str] = None
    data: Optional[str] = None
    action: Optional[str] = None
    source: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class MetaModel(BaseModel):
    source: Optional[str] = None
    client_ts: Optional[str] = None
    asr: Optional[Dict[str, Any]] = None
    voice_seconds: Optional[int] = None
    asr_duration_ms: Optional[int] = None


class IngestPayload(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    telegram_update_id: Optional[int] = None
    user: UserModel
    event: EventModel
    meta: Optional[MetaModel] = None


def get_settings() -> Settings:
    return Settings()


def inline_keyboard(options: List[List[Dict[str, str]]]) -> Dict[str, Any]:
    return {"inline": options}


def send_message_action(chat_id: int, text: str, keyboard: Any | None = None) -> Dict[str, Any]:
    return {
        "type": "send_message",
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "keyboard": keyboard,
    }


def diagnostic_question_payload(idx: int, total: int, chat_id: int, question: Dict[str, Any]) -> Dict[str, Any]:
    options_raw = question.get("options") or []
    options = []
    for opt_idx, opt in enumerate(options_raw):
        text = str(opt).strip()
        if not text:
            continue
        options.append([{"text": text, "data": f"diag:{idx}:{opt_idx}"}])
    return send_message_action(
        chat_id=chat_id,
        text=f"Диагностика, вопрос {idx + 1}/{total}:\n\n{question.get('text')}",
        keyboard=inline_keyboard(options),
    )


def training_case_payload(skill: str, idx: int, chat_id: int, cases: List[str]) -> Dict[str, Any]:
    if idx >= len(cases):
        return send_message_action(
            chat_id=chat_id,
            text="Кейсы закончились. Можем пройти заново или выбрать другой навык.",
            keyboard=inline_keyboard(
                [
                    [{"text": "С начала", "data": "training:restart"}],
                    [{"text": "Выбрать навык", "data": "action:skill:feedback"}],
                    [{"text": "Назад в меню", "data": "action:start"}],
                ]
            ),
        )
    return send_message_action(
        chat_id=chat_id,
        text=f"Кейс {idx + 1}:\n\n{cases[idx]}",
        keyboard=inline_keyboard(
            [
                [{"text": "Напомнить позже", "data": "remind:later"}],
                [{"text": "В меню", "data": "action:start"}],
                [{"text": "Выбрать навык", "data": "action:skill:feedback"}],
            ]
        ),
    )


async def schedule_reminder(settings: Settings, chat_id: int, token: Optional[str]) -> None:
    if not settings.frontend_push_url:
        logger.info("FRONTEND_PUSH_URL not set; skipping reminder scheduling")
        return

    await asyncio.sleep(settings.reminder_delay_seconds)
    payload = {
        "actions": [
            send_message_action(
                chat_id=chat_id,
                text="Напоминаю: у тебя незавершённый кейс. Продолжаем?",
                keyboard=inline_keyboard(
                    [
                        [{"text": "Продолжить", "data": "resume:yes"}],
                        [{"text": "Напомнить позже", "data": "remind:later"}],
                    ]
                ),
            )
        ]
    }
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = await http_client.post(settings.frontend_push_url, json=payload, headers=headers)
        resp.raise_for_status()
        logger.info("Reminder push sent to frontend for chat %s", chat_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to push reminder: %s", exc)


def progress_summary(state) -> str:
    diag = "пройдено" if state.diagnostic_done else "не завершена"
    return (
        f"Диагностика: {diag}\n"
        f"Навык: {'Обратная связь' if state.skill == 'feedback' else 'ИПР'}\n"
        f"Кейсов пройдено: {state.training_index}"
    )


async def ensure_diagnostic_questions(
    state, settings: Settings, ai: Optional[AIClient]
) -> List[Dict[str, Any]]:
    if ai and ai.enabled() and not state.diagnostic_questions:
        try:
            state.diagnostic_questions = await ai.generate_diagnostic(
                sphere=state.sphere,
                skill=state.skill,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI diagnostic generation failed: %s", exc)
            state.diagnostic_questions = []
    if not state.diagnostic_questions and not (ai and ai.enabled()):
        # Only fallback to static when AI недоступен
        state.diagnostic_questions = [
            {"text": q["text"], "options": [opt[0] for opt in q["options"]]}
            for q in DIAGNOSTIC_QUESTIONS
        ]
    return state.diagnostic_questions


async def ensure_training_cases(
    state, skill: str, settings: Settings, ai: Optional[AIClient]
) -> List[str]:
    if ai and ai.enabled() and not state.training_cases:
        try:
            state.training_cases = await ai.generate_cases(skill=skill, sphere=state.sphere)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI cases generation failed, fallback to defaults: %s", exc)
            state.training_cases = []
    if not state.training_cases:
        state.training_cases = (
            TRAINING_CASES_FEEDBACK if skill == "feedback" else TRAINING_CASES_IDP
        )
    return state.training_cases


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/conversations/{chat_id}")
async def get_conversations(chat_id: int) -> JSONResponse:
    """Get all conversations for a user."""
    conversations = ConversationDB.get_all_conversations(chat_id)
    return JSONResponse({"conversations": conversations})


@app.get("/conversations/{chat_id}/history")
async def get_conversation_history(chat_id: int, limit: int = 50) -> JSONResponse:
    """Get conversation history (messages) for a user."""
    messages = ConversationDB.get_conversation_history(chat_id, limit)
    return JSONResponse({"messages": messages})


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(f"ingest_requests {ingest_counter}")


# ==================== SKILL TRAINING ENDPOINTS ====================

@app.get("/skills/blocks")
async def get_skill_blocks() -> JSONResponse:
    """Get all skill blocks with basic info."""
    blocks = get_all_blocks()
    return JSONResponse({"blocks": blocks})


@app.get("/skills/blocks/{block_id}")
async def get_skill_block(block_id: str) -> JSONResponse:
    """Get a specific block with all its skills."""
    block = get_block(block_id)
    if not block:
        raise HTTPException(status_code=404, detail="Block not found")
    return JSONResponse({"block": block})


@app.get("/skills/blocks/{block_id}/skills/{skill_id}")
async def get_skill_detail(block_id: str, skill_id: str) -> JSONResponse:
    """Get a specific skill with its situations."""
    skill = get_skill(block_id, skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return JSONResponse({"skill": skill, "block_id": block_id})


class GenerateSituationRequest(BaseModel):
    chat_id: int
    block_id: str
    skill_id: str
    situation_index: Optional[int] = None


@app.post("/skills/generate-situation")
async def generate_situation(
    request: GenerateSituationRequest,
    settings: Settings = Depends(get_settings)
) -> JSONResponse:
    """Generate a training situation for a skill. Returns a situation for the user to respond to."""
    global ai_client
    if ai_client is None:
        ai_client = AIClient(settings, http_client)
    
    skill = get_skill(request.block_id, request.skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    
    # Get situation - either from predefined list or generate with AI
    situations = skill.get("situations", [])
    
    if request.situation_index is not None and 0 <= request.situation_index < len(situations):
        situation = situations[request.situation_index]
    elif situations:
        import random
        situation = random.choice(situations)
    else:
        # Generate with AI if no predefined situations
        if ai_client and ai_client.enabled():
            try:
                situation = await ai_client.generate_skill_situation(
                    skill_name=skill["name"],
                    skill_description=skill["description"],
                    theory_doc=skill.get("theory_doc", "")
                )
            except Exception as exc:
                logger.warning("AI situation generation failed: %s", exc)
                situation = f"Опишите ситуацию, в которой вам нужно применить навык '{skill['name']}'. Как бы вы действовали?"
        else:
            situation = f"Опишите ситуацию, в которой вам нужно применить навык '{skill['name']}'. Как бы вы действовали?"
    
    # Create session in database
    session_id = SkillTrainingDB.create_session(
        chat_id=request.chat_id,
        block_id=request.block_id,
        skill_id=request.skill_id,
        situation=situation
    )
    
    # Save to conversation history
    ConversationDB.get_or_create_user(request.chat_id)
    ConversationDB.save_message(
        chat_id=request.chat_id,
        role="assistant",
        content=situation,
        message_type="skill_situation",
        metadata={"session_id": session_id, "block_id": request.block_id, "skill_id": request.skill_id}
    )
    
    return JSONResponse({
        "session_id": session_id,
        "situation": situation,
        "skill": {
            "id": skill["id"],
            "name": skill["name"],
            "description": skill["description"]
        },
        "block_id": request.block_id
    })


class SubmitAnswerRequest(BaseModel):
    chat_id: int
    session_id: int
    answer: str


@app.post("/skills/submit-answer")
async def submit_answer(
    request: SubmitAnswerRequest,
    settings: Settings = Depends(get_settings)
) -> JSONResponse:
    """Submit user's answer to a skill situation and get AI feedback."""
    global ai_client
    if ai_client is None:
        ai_client = AIClient(settings, http_client)
    
    # Get session
    session = SkillTrainingDB.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session["chat_id"] != request.chat_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user")
    
    # Get skill info for context
    skill = get_skill(session["block_id"], session["skill_id"])
    skill_name = skill["name"] if skill else "Unknown"
    skill_description = skill["description"] if skill else ""
    theory_doc = skill.get("theory_doc", "") if skill else ""
    
    # Save user answer to conversation
    ConversationDB.save_message(
        chat_id=request.chat_id,
        role="user",
        content=request.answer,
        message_type="skill_answer",
        metadata={"session_id": request.session_id}
    )
    
    # Get AI feedback
    ai_feedback = None
    score = None
    
    if ai_client and ai_client.enabled():
        try:
            feedback_result = await ai_client.evaluate_skill_answer(
                skill_name=skill_name,
                skill_description=skill_description,
                situation=session["situation"],
                user_answer=request.answer,
                theory_doc=theory_doc
            )
            ai_feedback = feedback_result.get("feedback", "")
            score = feedback_result.get("score")
        except Exception as exc:
            logger.warning("AI feedback generation failed: %s", exc)
            ai_feedback = "Спасибо за ответ. К сожалению, не удалось получить оценку от AI. Попробуйте позже."
    else:
        # Simple heuristic feedback
        word_count = len(request.answer.split())
        if word_count < 20:
            ai_feedback = "Ответ слишком краткий. Попробуйте дать более развёрнутый ответ с конкретными примерами и действиями."
            score = 3
        elif word_count < 50:
            ai_feedback = "Неплохо, но можно добавить больше конкретики. Опишите конкретные шаги и ожидаемые результаты."
            score = 5
        else:
            ai_feedback = "Хороший развёрнутый ответ. Продолжайте практиковаться для закрепления навыка."
            score = 7
    
    # Save answer to database
    SkillTrainingDB.save_answer(
        session_id=request.session_id,
        chat_id=request.chat_id,
        user_answer=request.answer,
        ai_feedback=ai_feedback,
        score=score
    )
    
    # Mark session as completed
    SkillTrainingDB.complete_session(request.session_id)
    
    # Save feedback to conversation
    ConversationDB.save_message(
        chat_id=request.chat_id,
        role="assistant",
        content=ai_feedback,
        message_type="skill_feedback",
        metadata={"session_id": request.session_id, "score": score}
    )
    
    return JSONResponse({
        "session_id": request.session_id,
        "feedback": ai_feedback,
        "score": score,
        "status": "completed"
    })


@app.get("/skills/progress/{chat_id}")
async def get_skill_progress(chat_id: int) -> JSONResponse:
    """Get user's progress across all skills."""
    progress = SkillTrainingDB.get_user_progress(chat_id)
    sessions = SkillTrainingDB.get_user_sessions(chat_id)
    
    return JSONResponse({
        "progress": progress,
        "recent_sessions": sessions[:10]  # Last 10 sessions
    })


@app.get("/skills/sessions/{chat_id}")
async def get_user_skill_sessions(
    chat_id: int,
    block_id: Optional[str] = None,
    skill_id: Optional[str] = None
) -> JSONResponse:
    """Get all skill training sessions for a user."""
    sessions = SkillTrainingDB.get_user_sessions(chat_id, block_id, skill_id)
    return JSONResponse({"sessions": sessions})


@app.get("/skills/session/{session_id}")
async def get_skill_session_detail(session_id: int) -> JSONResponse:
    """Get details of a specific skill training session including answers."""
    session = SkillTrainingDB.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    answers = SkillTrainingDB.get_session_answers(session_id)
    skill = get_skill(session["block_id"], session["skill_id"])
    
    return JSONResponse({
        "session": session,
        "answers": answers,
        "skill": skill
    })


@app.post("/ingest")
async def ingest(payload: IngestPayload, settings: Settings = Depends(get_settings)) -> JSONResponse:
    global ingest_counter
    global ai_client
    if ai_client is None:
        ai_client = AIClient(settings, http_client)
    ingest_counter += 1
    user = payload.user
    event = payload.event
    meta = payload.meta
    state = await state_store.get(user.chat_id)

    # Save user to database
    ConversationDB.get_or_create_user(user.chat_id, user.user_id, user.username)

    # Save incoming message to database
    user_content = event.text or event.data or event.action or ""
    if user_content:
        ConversationDB.save_message(
            chat_id=user.chat_id,
            role="user",
            content=user_content,
            message_type=event.type,
            metadata={"action": event.action, "data": event.data}
        )

    actions: List[Dict[str, Any]] = []
    etype = (event.type or "").lower()
    data = event.data or ""
    action_name = event.action or ""

    if etype == "action":
        if action_name in ("start", "menu:start", "action:start"):
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Привет! Я помогу натренировать нужные навыки под твой запрос. Выбери, с чего начать.",
                    keyboard=inline_keyboard(
                        [
                            [{"text": "Начать диагностику", "data": "action:diagnostic:start"}],
                            [{"text": "Перейти к тренажеру", "data": "action:training:start"}],
                            [{"text": "Напоминания", "data": "action:menu:reminders"}],
                            [{"text": "Сфера деятельности", "data": "action:sphere:menu"}],
                        ]
                    ),
                )
            )
        elif action_name.startswith("sphere:menu"):
            await state_store.set_sphere_pending(user.chat_id, True)
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Напиши в чат свою специальность или сферу — введи её вручную.",
                    keyboard=inline_keyboard(
                        [
                            [{"text": "Отмена", "data": "action:start"}],
                        ]
                    ),
                )
            )
        elif action_name == "sphere:custom":
            await state_store.set_sphere_pending(user.chat_id, True)
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Напиши в чат свою специальность или сферу — введи её вручную.",
                    keyboard=inline_keyboard(
                        [
                            [{"text": "Отмена", "data": "action:start"}],
                        ]
                    ),
                )
            )
        elif action_name.startswith("sphere:"):
            sphere_code = action_name.split("sphere:", 1)[1]
            for code, label in SPHERES:
                if code == sphere_code:
                    await state_store.set_sphere(user.chat_id, code)
                    actions.append(
                        send_message_action(
                            chat_id=user.chat_id,
                            text=f"Сфера выбрана: {label}. Можно начинать диагностику или тренажёр.",
                            keyboard=inline_keyboard(
                                [
                                    [{"text": "Начать диагностику", "data": "action:diagnostic:start"}],
                                    [{"text": "Перейти к тренажёру", "data": "action:training:start"}],
                                    [{"text": "В меню", "data": "action:start"}],
                                ]
                            ),
                        )
                    )
                    break
            else:
                actions.append(
                    send_message_action(
                        chat_id=user.chat_id,
                        text="Неизвестная сфера. Выбери из списка.",
                        keyboard=inline_keyboard(
                            [[{"text": label, "data": f"action:sphere:{code}"}] for code, label in SPHERES]
                        ),
                    )
                )
        elif action_name == "diagnostic:start":
            await state_store.reset_diagnostic(user.chat_id)
            state = await state_store.get(user.chat_id)
            if not state.sphere_chosen:
                await state_store.set_sphere_pending(user.chat_id, True)
                actions.append(
                    send_message_action(
                        chat_id=user.chat_id,
                        text="Напиши в чат свою специальность или сферу — введи её вручную.",
                        keyboard=inline_keyboard(
                            [
                                [{"text": "Отмена", "data": "action:start"}],
                            ]
                        ),
                    )
                )
                return JSONResponse({"actions": actions})
            questions = await ensure_diagnostic_questions(state, settings, ai_client)
            if not questions:
                actions.append(
                    send_message_action(
                        chat_id=user.chat_id,
                        text="Не удалось получить вопросы диагностики от AI. Попробуйте позже.",
                        keyboard=inline_keyboard(
                            [
                                [{"text": "В меню", "data": "action:start"}],
                            ]
                        ),
                    )
                )
            else:
                actions.append(
                    diagnostic_question_payload(
                        0, len(questions), user.chat_id, questions[0]
                    )
                )
        elif action_name == "training:start":
            if not state.sphere_chosen:
                await state_store.set_sphere_pending(user.chat_id, True)
                actions.append(
                    send_message_action(
                        chat_id=user.chat_id,
                        text="Напиши в чат свою специальность или сферу — введи её вручную.",
                        keyboard=inline_keyboard(
                            [
                                [{"text": "Отмена", "data": "action:start"}],
                            ]
                        ),
                    )
                )
                return JSONResponse({"actions": actions})
            if not state.skill_chosen:
                await state_store.set_skill_pending(user.chat_id, True)
                actions.append(
                    send_message_action(
                        chat_id=user.chat_id,
                        text="Напиши, что хочешь потренировать (навык/тематику) — введи вручную.",
                        keyboard=inline_keyboard(
                            [
                                [{"text": "Отмена", "data": "action:start"}],
                            ]
                        ),
                    )
                )
                return JSONResponse({"actions": actions})
            # если уже есть незавершённый кейс — не дублируем новый, просто напомним
            if state.training_case_pending and state.training_cases:
                idx = state.training_index
                actions.append(training_case_payload(state.skill, idx, user.chat_id, state.training_cases))
                return JSONResponse({"actions": actions})
            await state_store.set_training_pending(user.chat_id, True)
            state.training_index = 0
            state.training_cases = []
            cases = await ensure_training_cases(state, state.skill, settings, ai_client)
            actions.append(training_case_payload(state.skill, 0, user.chat_id, cases))
            logger.info(
                "Start training skill=%s cases=%s", state.skill, [c[:80] for c in cases]
            )
        elif action_name == "menu:progress":
            # progress section removed
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Этот раздел временно недоступен.",
                    keyboard=inline_keyboard(
                        [
                            [{"text": "В меню", "data": "action:start"}],
                        ]
                    ),
                )
            )
        elif action_name == "menu:reminders":
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Напоминания включены по запросу. Хочешь получить напоминание позже?",
                    keyboard=inline_keyboard(
                        [
                            [{"text": "Да, напомни", "data": "remind:later"}],
                            [{"text": "Нет, продолжим", "data": "resume:yes"}],
                        ]
                    ),
                )
            )
        elif action_name == "skill:feedback":
            await state_store.set_skill_pending(user.chat_id, True)
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Напиши, что хочешь потренировать (навык/тематику) — введи вручную.",
                    keyboard=inline_keyboard(
                        [
                            [{"text": "Отмена", "data": "action:start"}],
                        ]
                    ),
                )
            )
        elif action_name == "skill:idp":
            await state_store.set_skill_pending(user.chat_id, True)
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Напиши, что хочешь потренировать (навык/тематику) — введи вручную.",
                    keyboard=inline_keyboard(
                        [
                            [{"text": "Отмена", "data": "action:start"}],
                        ]
                    ),
                )
            )
        elif action_name == "menu:toc":
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Этот раздел временно недоступен.",
                    keyboard=inline_keyboard([[{"text": "В меню", "data": "action:start"}]]),
                )
            )
        else:
            actions.append(send_message_action(chat_id=user.chat_id, text="Команда принята."))

    elif etype == "callback":
        if data.startswith("diag:"):
            state.diagnostic_answers.append(data)
            next_idx = len(state.diagnostic_answers)
            questions = state.diagnostic_questions or [
                {"text": q["text"], "options": [opt[0] for opt in q["options"]]}
                for q in DIAGNOSTIC_QUESTIONS
            ]
            if next_idx < len(questions):
                actions.append(
                    diagnostic_question_payload(
                        next_idx, len(questions), user.chat_id, questions[next_idx]
                    )
                )
            else:
                state.diagnostic_done = True
                summary = interpret_diagnostic(state.diagnostic_answers)
                if ai_client and ai_client.enabled():
                    try:
                        summary = await ai_client.summarize_diagnostic(
                            questions=questions,
                            answers=state.diagnostic_answers,
                            sphere=state.sphere,
                            skill=state.skill,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("AI diagnostic summary failed, fallback to heuristic: %s", exc)
                actions.append(
                    send_message_action(
                        chat_id=user.chat_id,
                        text=f"Диагностика завершена.\n\n{summary}",
                        keyboard=inline_keyboard(
                            [
                                [{"text": "Перейти к тренажёру", "data": "action:training:start"}],
                                [{"text": "Ещё раз диагностику", "data": "action:diagnostic:start"}],
                                [{"text": "В меню", "data": "action:start"}],
                            ]
                        ),
                    )
                )
        elif data.startswith("case:next") or data == "resume:yes":
            await state_store.set_training_pending(user.chat_id, True)
            next_idx = await state_store.increment_training(user.chat_id)
            cases = await ensure_training_cases(state, state.skill, settings, ai_client)
            actions.append(training_case_payload(state.skill, next_idx, user.chat_id, cases))
        elif data.startswith("training:restart"):
            state.training_index = 0
            await state_store.set_training_pending(user.chat_id, True)
            cases = await ensure_training_cases(state, state.skill, settings, ai_client)
            actions.append(training_case_payload(state.skill, 0, user.chat_id, cases))
        elif data.startswith("remind:later"):
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Хорошо, напомню позже.",
                )
            )
            asyncio.create_task(
                schedule_reminder(settings, user.chat_id, settings.frontend_push_token)
            )
        elif data.startswith("case:retry"):
            await state_store.set_training_pending(user.chat_id, True)
            cases = await ensure_training_cases(state, state.skill, settings, ai_client)
            actions.append(
                training_case_payload(state.skill, state.training_index, user.chat_id, cases)
            )
        else:
            actions.append(send_message_action(chat_id=user.chat_id, text="Сигнал получен."))

    elif etype == "text":
        if state.sphere_pending and not state.sphere_chosen:
            # treat this text as sphere selection
            await state_store.set_sphere(user.chat_id, event.text or "general")
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text=f"Сфера установлена: {event.text}. Что дальше?",
                    keyboard=inline_keyboard(
                        [
                            [{"text": "Начать диагностику", "data": "action:diagnostic:start"}],
                            [{"text": "Перейти к тренажёру", "data": "action:training:start"}],
                            [{"text": "В меню", "data": "action:start"}],
                        ]
                    ),
                )
            )
            return JSONResponse({"actions": actions})
        if state.skill_pending and not state.skill_chosen:
            skill_text = (event.text or "").strip() or "другой навык"
            await state_store.set_skill(user.chat_id, skill_text)
            state = await state_store.get(user.chat_id)
            await state_store.set_training_pending(user.chat_id, True)
            state.training_index = 0
            state.training_cases = []
            cases = await ensure_training_cases(state, state.skill, settings, ai_client)
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Готовлю тренинг, собираю первый кейс...",
                )
            )
            if cases:
                actions.append(training_case_payload(state.skill, 0, user.chat_id, cases))
            else:
                actions.append(
                    send_message_action(
                        chat_id=user.chat_id,
                        text="Не удалось подготовить кейсы для этой темы. Попробуй снова или выбери другую.",
                        keyboard=inline_keyboard(
                            [
                                [{"text": "Отмена", "data": "action:start"}],
                            ]
                        ),
                    )
                )
            logger.info("Start training after custom skill skill=%s cases=%s", state.skill, [c[:80] for c in cases])
            return JSONResponse({"actions": actions})

        if state.training_case_pending:
            case_idx = state.training_index
            cases = await ensure_training_cases(state, state.skill, settings, ai_client)
            case_text = (
                cases[case_idx]
                if case_idx < len(cases)
                else get_case(state.skill, case_idx)
            )
            if ai_client and ai_client.enabled():
                try:
                    ai_actions = await ai_client.build_actions(
                        skill=state.skill,
                        sphere=state.sphere,
                        case_text=case_text,
                        user_answer=event.text or "",
                    )
                    # ensure chat_id is set and track if AI считает ответ ок
                    ai_says_next = False
                    for act in ai_actions:
                        act["chat_id"] = user.chat_id
                        kb = act.get("keyboard", {}) or {}
                        inline = kb.get("inline") or []
                        # always add navigation buttons
                        inline = inline + [
                            [{"text": "В меню", "data": "action:start"}],
                            [{"text": "Выбрать навык", "data": "action:skill:feedback"}],
                        ]
                        act["keyboard"] = {"inline": inline}
                        for row in inline:
                            for btn in row:
                                if (btn.get("data") or "").startswith("case:next"):
                                    ai_says_next = True
                        logger.info("AI action: %s", act)
                    actions.extend(ai_actions)
                    if ai_says_next:
                        await state_store.set_training_pending(user.chat_id, False)
                    if not ai_actions:
                        raise RuntimeError("AI returned no actions")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("AI client failed, fallback to heuristic: %s", exc)
                    good, feedback = evaluate_answer(event.text or "", settings)
                    actions.append(
                        send_message_action(
                            chat_id=user.chat_id,
                            text=f"{feedback}",
                            keyboard=inline_keyboard(
                                [[{"text": "Дальше", "data": "case:next"}]]
                                if good
                                else [[{"text": "Попробовать снова", "data": "case:retry"}]]
                            ),
                        )
                    )
                    if good:
                        await state_store.set_training_pending(user.chat_id, False)
            else:
                good, feedback = evaluate_answer(event.text or "", settings)
                actions.append(
                    send_message_action(
                        chat_id=user.chat_id,
                        text=f"{feedback}",
                        keyboard=inline_keyboard(
                            [[{"text": "Дальше", "data": "case:next"}]]
                            if good
                            else [
                                [{"text": "Попробовать снова", "data": "case:retry"}],
                                [{"text": "В меню", "data": "action:start"}],
                            ]
                        ),
                    )
                )
                if good:
                    await state_store.set_training_pending(user.chat_id, False)
        else:
            actions.append(
                send_message_action(
                    chat_id=user.chat_id,
                    text="Принял сообщение. Чтобы продолжить тренировку, выбери действие в меню.",
                    keyboard=inline_keyboard(
                        [
                            [{"text": "Перейти к тренажёру", "data": "action:training:start"}],
                            [{"text": "Прогресс", "data": "action:menu:progress"}],
                            [{"text": "В меню", "data": "action:start"}],
                        ]
                    ),
                )
            )
    else:
        actions.append(send_message_action(chat_id=user.chat_id, text="Событие принято."))

    # Save bot responses to database
    for act in actions:
        if act.get("type") == "send_message" and act.get("text"):
            ConversationDB.save_message(
                chat_id=user.chat_id,
                role="assistant",
                content=act.get("text", ""),
                message_type="send_message",
                metadata={"keyboard": act.get("keyboard")}
            )

    return JSONResponse({"actions": dedup_actions(actions)})


@app.middleware("http")
async def add_timeout_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Backend"] = "training-bot"
    return response


def dedup_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_text: dict[tuple[str, str], Dict[str, Any]] = {}
    for act in actions:
        if act.get("type") != "send_message":
            # keep non-send_message actions as is
            best_by_text.setdefault(("__other__", str(len(best_by_text))), act)
            continue
        text = str(act.get("text") or "")
        parse_mode = str(act.get("parse_mode") or "HTML")
        keyboard_raw = (
            act.get("keyboard", {}).get("inline")
            if isinstance(act.get("keyboard"), dict)
            else act.get("keyboard")
        )
        key = (text, parse_mode)
        existing = best_by_text.get(key)
        has_keyboard = bool(keyboard_raw)
        if existing is None or (has_keyboard and not existing.get("__has_keyboard")):
            copy = dict(act)
            copy["__has_keyboard"] = has_keyboard
            best_by_text[key] = copy
    result: List[Dict[str, Any]] = []
    for key, act in best_by_text.items():
        act.pop("__has_keyboard", None)
        # skip placeholder keys for non-send_message
        if key[0] == "__other__":
            result.append(act)
        else:
            result.append(act)
    return result


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
