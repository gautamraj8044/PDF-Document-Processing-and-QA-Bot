from __future__ import annotations

import uuid
from dataclasses import dataclass

import psycopg


@dataclass
class User:
    id: str
    email: str
    hashed_password: str


class UserStore:
    def __init__(self, postgres_url: str) -> None:
        self._url = postgres_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._url, autocommit=True)

    def setup(self) -> None:
        """Create users table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

    def create_user(self, email: str, hashed_password: str) -> User:
        user_id = str(uuid.uuid4())
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (id, email, hashed_password) VALUES (%s, %s, %s)",
                    (user_id, email.lower().strip(), hashed_password),
                )
            except psycopg.errors.UniqueViolation:
                raise ValueError("An account with this email already exists.")
        return User(id=user_id, email=email, hashed_password=hashed_password)

    def get_by_email(self, email: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, hashed_password FROM users WHERE email = %s",
                (email.lower().strip(),),
            ).fetchone()
        if not row:
            return None
        return User(id=row[0], email=row[1], hashed_password=row[2])

    def get_by_id(self, user_id: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, hashed_password FROM users WHERE id = %s",
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return User(id=row[0], email=row[1], hashed_password=row[2])