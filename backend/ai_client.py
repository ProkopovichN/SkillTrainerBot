from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from config import Settings


logger = logging.getLogger(__name__)


class AIClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    def enabled(self) -> bool:
        return bool(self.settings.openrouter_api_key)

    async def build_actions(
        self,
        skill: str,
        case_text: str,
        user_answer: str,
        sphere: str,
    ) -> List[Dict[str, Any]]:
        """
        Ask the model to score the answer and return JSON actions.
        """
        if not self.enabled():
            raise RuntimeError("OpenRouter API key not configured")

        prompt = self._prompt(skill=skill, sphere=sphere, case_text=case_text, user_answer=user_answer)
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.settings.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты тренер по навыкам обратной связи и ИПР. "
                        "Всегда отвечай JSON без текста вокруг. "
                        'Формат: {"actions":[{"type":"send_message","chat_id":<int>,"text":"...","parse_mode":"HTML","keyboard":{"inline":[[{"text":"...","data":"..."}]]}}]} '
                        "Текст должен быть кратким, на русском, без Markdown, только HTML (b, i, code, ul/li). "
                        "Если ответ ок — предложи кнопку 'Дальше' (data: case:next). "
                        "Если слабый — кнопка 'Попробовать снова' (data: case:retry) и короткая подсказка."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.settings.openrouter_temperature,
        }
        resp = await self.client.post(
            self.settings.openrouter_base_url, headers=headers, json=body
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        logger.info("AI raw content: %s", content)
        try:
            actions = json.loads(content).get("actions", [])
            if not actions:
                raise RuntimeError("empty actions")
            return actions
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"AI response is not valid JSON or empty: {content}") from exc

    async def generate_diagnostic(
        self,
        num_questions: int = 3,
        options_per_question: int = 3,
        sphere: str | None = None,
        skill: str | None = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled():
            raise RuntimeError("OpenRouter API key not configured")

        prompt = (
            "Сформируй краткую диагностику по выбранному навыку. "
            f"Сфера: {sphere or 'общая'}. Навык/тематика: {skill or 'обратная связь и ИПР'}. "
            f"Нужно {num_questions} вопросов. Формат строго JSON без пояснений: "
            '{"questions":[{"question":"...","options":["opt1","opt2","opt3"]},...]} '
            f"Каждый вопрос должен иметь ровно {options_per_question} варианта ответа. "
            "Кратко, по-деловому, на русском."
        )
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.openrouter_temperature,
        }
        resp = await self.client.post(self.settings.openrouter_base_url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        try:
            # some providers may already return dict instead of string
            parsed = json.loads(content) if isinstance(content, str) else content
            questions = parsed.get("questions", []) if isinstance(parsed, dict) else []
            # normalize
            result = []
            for q in questions:
                question_text = str(q.get("question") or "").strip()
                opts = q.get("options") or []
                opts = [str(o).strip() for o in opts if str(o).strip()]
                if not question_text or not opts:
                    continue
                result.append({"text": question_text, "options": opts[:options_per_question]})
            return result[:num_questions]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"AI diagnostic response invalid: {content}") from exc

    async def generate_cases(self, skill: str, sphere: str, num_cases: int = 2) -> List[str]:
        if not self.enabled():
            raise RuntimeError("OpenRouter API key not configured")
        prompt = (
            f"Сформулируй {num_cases} кейса для тренажёра навыка '{skill}' в сфере '{sphere}'. "
            "Каждый кейс — краткое описание ситуации (1-2 предложения). "
            "Ответ верни строгим JSON: {\"cases\": [\"кейс1\", \"кейс2\"]} без пояснений."
        )
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.openrouter_temperature,
        }
        resp = await self.client.post(self.settings.openrouter_base_url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        try:
            cases = json.loads(content).get("cases", [])
            return [str(c).strip() for c in cases if str(c).strip()][:num_cases]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"AI cases response invalid: {content}") from exc

    async def summarize_diagnostic(
        self,
        questions: List[Dict[str, Any]],
        answers: List[str],
        sphere: str,
        skill: str,
    ) -> str:
        if not self.enabled():
            raise RuntimeError("OpenRouter API key not configured")
        q_lines = []
        for idx, q in enumerate(questions):
            opts = q.get("options") or []
            chosen_idx = None
            if idx < len(answers):
                try:
                    parts = answers[idx].split(":")
                    chosen_idx = int(parts[-1])
                except Exception:
                    chosen_idx = None
            chosen = opts[chosen_idx] if chosen_idx is not None and chosen_idx < len(opts) else ""
            q_lines.append(f"Q{idx+1}: {q.get('text')} | выбран: {chosen}")
        prompt = (
            "Подведи итоги диагностики.\n"
            f"Сфера: {sphere}\nНавык/тематика: {skill}\n"
            "Вопросы и выбранные ответы:\n" + "\n".join(q_lines) + "\n"
            "Сформулируй краткий вывод и совет следующего шага. Ответ верни текстом на русском, 1-2 абзаца."
        )
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.openrouter_temperature,
        }
        resp = await self.client.post(self.settings.openrouter_base_url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return str(content).strip()

    def _prompt(self, skill: str, sphere: str, case_text: str, user_answer: str) -> str:
        return (
            f"Сфера: {sphere}\n"
            f"Навык: {skill}\n"
            f"Кейс: {case_text}\n"
            f"Ответ пользователя: {user_answer}\n"
            "Оцени ответ и верни JSON с действиями (см. формат)."
        )
