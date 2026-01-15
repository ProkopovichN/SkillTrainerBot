import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UserProgress:
    diagnostic_answers: List[str] = field(default_factory=list)
    diagnostic_done: bool = False
    training_index: int = 0
    training_case_pending: bool = False
    skill: str = "feedback"  # or "idp"
    skill_chosen: bool = False
    skill_pending: bool = False
    sphere: str = "general"
    sphere_chosen: bool = False
    sphere_pending: bool = False
    last_reminder: Optional[str] = None
    diagnostic_questions: List[Dict[str, Any]] = field(default_factory=list)
    training_cases: List[str] = field(default_factory=list)


class StateStore:
    def __init__(self) -> None:
        self._store: Dict[int, UserProgress] = {}
        self._lock = asyncio.Lock()

    async def get(self, chat_id: int) -> UserProgress:
        async with self._lock:
            if chat_id not in self._store:
                self._store[chat_id] = UserProgress()
            return self._store[chat_id]

    async def reset_diagnostic(self, chat_id: int) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.diagnostic_answers = []
            state.diagnostic_done = False
            state.diagnostic_questions = []

    async def increment_training(self, chat_id: int) -> int:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.training_index += 1
            state.training_case_pending = True
            return state.training_index

    async def set_skill(self, chat_id: int, skill: str) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.skill = skill
            state.skill_chosen = True
            state.skill_pending = False
            state.training_cases = []
            state.training_index = 0
            state.training_case_pending = False

    async def set_skill_pending(self, chat_id: int, pending: bool) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.skill_pending = pending

    async def set_sphere(self, chat_id: int, sphere: str) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.sphere = sphere
            state.sphere_chosen = True
            state.sphere_pending = False
            state.diagnostic_questions = []

    async def set_sphere_pending(self, chat_id: int, pending: bool) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.sphere_pending = pending
            if pending:
                state.sphere_chosen = False
                state.training_case_pending = False
                state.training_index = 0
                state.training_cases = []

    async def set_training_pending(self, chat_id: int, pending: bool) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.training_case_pending = pending
