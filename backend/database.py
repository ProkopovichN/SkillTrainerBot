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
        
        # Skill training sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                block_id TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                situation TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES users(chat_id)
            )
        """)
        
        # Skill training answers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                user_answer TEXT NOT NULL,
                ai_feedback TEXT,
                score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES skill_sessions(id),
                FOREIGN KEY (chat_id) REFERENCES users(chat_id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_chat_id ON conversations(chat_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skill_sessions_chat_id ON skill_sessions(chat_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skill_answers_session_id ON skill_answers(session_id)")
        
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


class SkillTrainingDB:
    """Database operations for skill training sessions."""
    
    @staticmethod
    def create_session(chat_id: int, block_id: str, skill_id: str, situation: str) -> int:
        """Create a new skill training session."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO skill_sessions (chat_id, block_id, skill_id, situation, status)
                   VALUES (?, ?, ?, ?, 'pending')""",
                (chat_id, block_id, skill_id, situation)
            )
            return cursor.lastrowid
    
    @staticmethod
    def get_session(session_id: int) -> Optional[Dict[str, Any]]:
        """Get a skill training session by ID."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM skill_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    @staticmethod
    def get_pending_session(chat_id: int) -> Optional[Dict[str, Any]]:
        """Get the current pending session for a user."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM skill_sessions WHERE chat_id = ? AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
                (chat_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    @staticmethod
    def save_answer(session_id: int, chat_id: int, user_answer: str, ai_feedback: str = None, score: int = None) -> int:
        """Save user's answer to a skill training session."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO skill_answers (session_id, chat_id, user_answer, ai_feedback, score)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, chat_id, user_answer, ai_feedback, score)
            )
            return cursor.lastrowid
    
    @staticmethod
    def complete_session(session_id: int):
        """Mark a session as completed."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE skill_sessions SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,)
            )
    
    @staticmethod
    def get_user_sessions(chat_id: int, block_id: str = None, skill_id: str = None) -> List[Dict[str, Any]]:
        """Get all sessions for a user, optionally filtered by block/skill."""
        with get_db() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM skill_sessions WHERE chat_id = ?"
            params = [chat_id]
            
            if block_id:
                query += " AND block_id = ?"
                params.append(block_id)
            if skill_id:
                query += " AND skill_id = ?"
                params.append(skill_id)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def get_session_answers(session_id: int) -> List[Dict[str, Any]]:
        """Get all answers for a session."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM skill_answers WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def get_user_progress(chat_id: int) -> Dict[str, Any]:
        """Get user's overall progress across all skills."""
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Total sessions
            cursor.execute(
                "SELECT COUNT(*) as total FROM skill_sessions WHERE chat_id = ?",
                (chat_id,)
            )
            total = cursor.fetchone()["total"]
            
            # Completed sessions
            cursor.execute(
                "SELECT COUNT(*) as completed FROM skill_sessions WHERE chat_id = ? AND status = 'completed'",
                (chat_id,)
            )
            completed = cursor.fetchone()["completed"]
            
            # Sessions by block
            cursor.execute(
                """SELECT block_id, COUNT(*) as count, 
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
                   FROM skill_sessions WHERE chat_id = ? GROUP BY block_id""",
                (chat_id,)
            )
            by_block = [dict(row) for row in cursor.fetchall()]
            
            # Average score
            cursor.execute(
                "SELECT AVG(score) as avg_score FROM skill_answers WHERE chat_id = ? AND score IS NOT NULL",
                (chat_id,)
            )
            avg_score = cursor.fetchone()["avg_score"]
            
            return {
                "total_sessions": total,
                "completed_sessions": completed,
                "by_block": by_block,
                "average_score": round(avg_score, 2) if avg_score else None
            }


# Initialize database on module import
init_db()
