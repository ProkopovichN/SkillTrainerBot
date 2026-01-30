from __future__ import annotations

import sqlite3
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DATABASE_PATH = "conversations.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database tables."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER UNIQUE NOT NULL,
                user_id INTEGER,
                username TEXT,
                sphere TEXT DEFAULT 'general',
                skill TEXT DEFAULT 'feedback',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                session_type TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                status TEXT DEFAULT 'active',
                FOREIGN KEY (chat_id) REFERENCES users(chat_id)
            )
        """)
        
        # Messages table - stores all conversation messages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                message_type TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id),
                FOREIGN KEY (chat_id) REFERENCES users(chat_id)
            )
        """)
        
        # User progress table - persists training state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER UNIQUE NOT NULL,
                diagnostic_answers TEXT DEFAULT '[]',
                diagnostic_done INTEGER DEFAULT 0,
                training_index INTEGER DEFAULT 0,
                training_case_pending INTEGER DEFAULT 0,
                skill TEXT DEFAULT 'feedback',
                skill_chosen INTEGER DEFAULT 0,
                skill_pending INTEGER DEFAULT 0,
                sphere TEXT DEFAULT 'general',
                sphere_chosen INTEGER DEFAULT 0,
                sphere_pending INTEGER DEFAULT 0,
                diagnostic_questions TEXT DEFAULT '[]',
                training_cases TEXT DEFAULT '[]',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES users(chat_id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_chat_id ON conversations(chat_id)")
        
        logger.info("Database initialized successfully")


class ConversationDB:
    """Database operations for conversations."""
    
    @staticmethod
    def get_or_create_user(chat_id: int, user_id: int = None, username: str = None) -> Dict[str, Any]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE chat_id = ?", (chat_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            
            cursor.execute(
                "INSERT INTO users (chat_id, user_id, username) VALUES (?, ?, ?)",
                (chat_id, user_id, username)
            )
            return {"chat_id": chat_id, "user_id": user_id, "username": username}
    
    @staticmethod
    def update_user(chat_id: int, sphere: str = None, skill: str = None):
        with get_db() as conn:
            cursor = conn.cursor()
            updates = []
            params = []
            
            if sphere:
                updates.append("sphere = ?")
                params.append(sphere)
            if skill:
                updates.append("skill = ?")
                params.append(skill)
            
            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                params.append(chat_id)
                cursor.execute(
                    f"UPDATE users SET {', '.join(updates)} WHERE chat_id = ?",
                    params
                )
    
    @staticmethod
    def start_conversation(chat_id: int, session_type: str = "training") -> int:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (chat_id, session_type) VALUES (?, ?)",
                (chat_id, session_type)
            )
            return cursor.lastrowid
    
    @staticmethod
    def end_conversation(conversation_id: int):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE conversations SET ended_at = CURRENT_TIMESTAMP, status = 'completed' WHERE id = ?",
                (conversation_id,)
            )
    
    @staticmethod
    def get_active_conversation(chat_id: int) -> Optional[int]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM conversations WHERE chat_id = ? AND status = 'active' ORDER BY started_at DESC LIMIT 1",
                (chat_id,)
            )
            row = cursor.fetchone()
            return row["id"] if row else None
    
    @staticmethod
    def save_message(
        chat_id: int,
        role: str,
        content: str,
        message_type: str = None,
        metadata: Dict[str, Any] = None,
        conversation_id: int = None
    ) -> int:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get or create active conversation
            if not conversation_id:
                cursor.execute(
                    "SELECT id FROM conversations WHERE chat_id = ? AND status = 'active' ORDER BY started_at DESC LIMIT 1",
                    (chat_id,)
                )
                row = cursor.fetchone()
                if row:
                    conversation_id = row["id"]
                else:
                    cursor.execute(
                        "INSERT INTO conversations (chat_id, session_type) VALUES (?, ?)",
                        (chat_id, "general")
                    )
                    conversation_id = cursor.lastrowid
            
            cursor.execute(
                """INSERT INTO messages (conversation_id, chat_id, role, content, message_type, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conversation_id, chat_id, role, content, message_type, json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid
    
    @staticmethod
    def get_conversation_history(chat_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT role, content, message_type, metadata, created_at 
                   FROM messages 
                   WHERE chat_id = ? 
                   ORDER BY created_at DESC 
                   LIMIT ?""",
                (chat_id, limit)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]
    
    @staticmethod
    def get_all_conversations(chat_id: int) -> List[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT c.id, c.session_type, c.started_at, c.ended_at, c.status,
                          COUNT(m.id) as message_count
                   FROM conversations c
                   LEFT JOIN messages m ON c.id = m.conversation_id
                   WHERE c.chat_id = ?
                   GROUP BY c.id
                   ORDER BY c.started_at DESC""",
                (chat_id,)
            )
            return [dict(row) for row in cursor.fetchall()]


class ProgressDB:
    """Database operations for user progress (persistent state)."""
    
    @staticmethod
    def get_progress(chat_id: int) -> Optional[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_progress WHERE chat_id = ?", (chat_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data["diagnostic_answers"] = json.loads(data["diagnostic_answers"] or "[]")
                data["diagnostic_questions"] = json.loads(data["diagnostic_questions"] or "[]")
                data["training_cases"] = json.loads(data["training_cases"] or "[]")
                return data
            return None
    
    @staticmethod
    def save_progress(chat_id: int, progress: Dict[str, Any]):
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Check if exists
            cursor.execute("SELECT id FROM user_progress WHERE chat_id = ?", (chat_id,))
            exists = cursor.fetchone()
            
            diagnostic_answers = json.dumps(progress.get("diagnostic_answers", []))
            diagnostic_questions = json.dumps(progress.get("diagnostic_questions", []))
            training_cases = json.dumps(progress.get("training_cases", []))
            
            if exists:
                cursor.execute("""
                    UPDATE user_progress SET
                        diagnostic_answers = ?,
                        diagnostic_done = ?,
                        training_index = ?,
                        training_case_pending = ?,
                        skill = ?,
                        skill_chosen = ?,
                        skill_pending = ?,
                        sphere = ?,
                        sphere_chosen = ?,
                        sphere_pending = ?,
                        diagnostic_questions = ?,
                        training_cases = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE chat_id = ?
                """, (
                    diagnostic_answers,
                    int(progress.get("diagnostic_done", False)),
                    progress.get("training_index", 0),
                    int(progress.get("training_case_pending", False)),
                    progress.get("skill", "feedback"),
                    int(progress.get("skill_chosen", False)),
                    int(progress.get("skill_pending", False)),
                    progress.get("sphere", "general"),
                    int(progress.get("sphere_chosen", False)),
                    int(progress.get("sphere_pending", False)),
                    diagnostic_questions,
                    training_cases,
                    chat_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO user_progress (
                        chat_id, diagnostic_answers, diagnostic_done, training_index,
                        training_case_pending, skill, skill_chosen, skill_pending,
                        sphere, sphere_chosen, sphere_pending, diagnostic_questions, training_cases
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chat_id,
                    diagnostic_answers,
                    int(progress.get("diagnostic_done", False)),
                    progress.get("training_index", 0),
                    int(progress.get("training_case_pending", False)),
                    progress.get("skill", "feedback"),
                    int(progress.get("skill_chosen", False)),
                    int(progress.get("skill_pending", False)),
                    progress.get("sphere", "general"),
                    int(progress.get("sphere_chosen", False)),
                    int(progress.get("sphere_pending", False)),
                    diagnostic_questions,
                    training_cases
                ))


# Initialize database on module import
init_db()
