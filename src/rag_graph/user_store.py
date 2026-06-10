from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from .models import Base, UserModel


@dataclass(slots=True)
class User:
    id: str
    email: str
    hashed_password: str


class UserStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._engine = create_engine(
            database_url,
            connect_args=self._connect_args(database_url),
            future=True,
        )
        self._session_factory = sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            future=True,
        )

    def setup(self) -> None:
        Base.metadata.create_all(self._engine)

    def close(self) -> None:
        self._engine.dispose()

    def create_user(self, email: str, hashed_password: str) -> User:
        normalized_email = _normalize_email(email)
        user_id = _new_user_id()

        with self._session_factory() as session:
            existing = session.scalar(
                select(UserModel).where(UserModel.email == normalized_email)
            )
            if existing:
                raise ValueError("An account with this email already exists.")

            session.add(
                UserModel(
                    id=user_id,
                    email=normalized_email,
                    hashed_password=hashed_password,
                )
            )
            session.commit()

        return User(id=user_id, email=normalized_email, hashed_password=hashed_password)

    def get_by_email(self, email: str) -> User | None:
        normalized_email = _normalize_email(email)
        with self._session_factory() as session:
            model = session.scalar(
                select(UserModel).where(UserModel.email == normalized_email)
            )
            if model is None:
                return None
            return User(id=model.id, email=model.email, hashed_password=model.hashed_password)

    def get_by_id(self, user_id: str) -> User | None:
        with self._session_factory() as session:
            model = session.get(UserModel, user_id)
            if model is None:
                return None
            return User(id=model.id, email=model.email, hashed_password=model.hashed_password)

    @staticmethod
    def _connect_args(database_url: str) -> dict[str, object]:
        parsed = make_url(database_url)
        if not parsed.drivername.startswith("sqlite"):
            return {}

        database = parsed.database
        if database and database != ":memory:":
            path = Path(database)
            if not path.is_absolute():
                path = Path.cwd() / path
            path.parent.mkdir(parents=True, exist_ok=True)

        return {"check_same_thread": False}


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _new_user_id() -> str:
    import uuid

    return str(uuid.uuid4())
