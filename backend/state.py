import asyncio
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from database import ProgressDB


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProgress":
        return cls(
            diagnostic_answers=data.get("diagnostic_answers", []),
            diagnostic_done=bool(data.get("diagnostic_done", False)),
            training_index=data.get("training_index", 0),
            training_case_pending=bool(data.get("training_case_pending", False)),
            skill=data.get("skill", "feedback"),
            skill_chosen=bool(data.get("skill_chosen", False)),
            skill_pending=bool(data.get("skill_pending", False)),
            sphere=data.get("sphere", "general"),
            sphere_chosen=bool(data.get("sphere_chosen", False)),
            sphere_pending=bool(data.get("sphere_pending", False)),
            last_reminder=data.get("last_reminder"),
            diagnostic_questions=data.get("diagnostic_questions", []),
            training_cases=data.get("training_cases", []),
        )


class StateStore:
    def __init__(self) -> None:
        self._store: Dict[int, UserProgress] = {}
        self._lock = asyncio.Lock()

    async def _save_to_db(self, chat_id: int, state: UserProgress) -> None:
        """Persist state to database."""
        ProgressDB.save_progress(chat_id, state.to_dict())

    async def _load_from_db(self, chat_id: int) -> Optional[UserProgress]:
        """Load state from database."""
        data = ProgressDB.get_progress(chat_id)
        if data:
            return UserProgress.from_dict(data)
        return None

    async def get(self, chat_id: int) -> UserProgress:
        async with self._lock:
            if chat_id not in self._store:
                # Try to load from database first
                db_state = await self._load_from_db(chat_id)
                if db_state:
                    self._store[chat_id] = db_state
                else:
                    self._store[chat_id] = UserProgress()
            return self._store[chat_id]

    async def reset_diagnostic(self, chat_id: int) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.diagnostic_answers = []
            state.diagnostic_done = False
            state.diagnostic_questions = []
            await self._save_to_db(chat_id, state)

    async def increment_training(self, chat_id: int) -> int:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.training_index += 1
            state.training_case_pending = True
            await self._save_to_db(chat_id, state)
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
            await self._save_to_db(chat_id, state)

    async def set_skill_pending(self, chat_id: int, pending: bool) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.skill_pending = pending
            await self._save_to_db(chat_id, state)

    async def set_sphere(self, chat_id: int, sphere: str) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.sphere = sphere
            state.sphere_chosen = True
            state.sphere_pending = False
            state.diagnostic_questions = []
            await self._save_to_db(chat_id, state)

    async def set_sphere_pending(self, chat_id: int, pending: bool) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.sphere_pending = pending
            if pending:
                state.sphere_chosen = False
                state.training_case_pending = False
                state.training_index = 0
                state.training_cases = []
            await self._save_to_db(chat_id, state)

    async def set_training_pending(self, chat_id: int, pending: bool) -> None:
        async with self._lock:
            state = self._store.setdefault(chat_id, UserProgress())
            state.training_case_pending = pending
            await self._save_to_db(chat_id, state)
