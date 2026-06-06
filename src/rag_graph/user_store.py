from __future__ import annotations

import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .models import Base, UserModel


class User:
    def __init__(self, id: str, email: str, hashed_password: str) -> None:
        self.id = id
        self.email = email
        self.hashed_password = hashed_password


class UserStore:
    def __init__(self, postgres_url: str) -> None:
        self._engine = create_engine(postgres_url)

    def setup(self) -> None:
        Base.metadata.create_all(self._engine)

    def create_user(self, email: str, hashed_password: str) -> User:
        user_id = str(uuid.uuid4())
        with Session(self._engine) as session:
            existing = session.query(UserModel).filter(
                UserModel.email == email.lower().strip()
            ).first()
            if existing:
                raise ValueError("An account with this email already exists.")
            model = UserModel(
                id=user_id,
                email=email.lower().strip(),
                hashed_password=hashed_password,
            )
            session.add(model)
            session.commit()
        return User(id=user_id, email=email, hashed_password=hashed_password)

    def get_by_email(self, email: str) -> User | None:
        with Session(self._engine) as session:
            model = session.query(UserModel).filter(
                UserModel.email == email.lower().strip()
            ).first()
            if not model:
                return None
            return User(id=model.id, email=model.email, hashed_password=model.hashed_password)

    def get_by_id(self, user_id: str) -> User | None:
        with Session(self._engine) as session:
            model = session.get(UserModel, user_id)
            if not model:
                return None
            return User(id=model.id, email=model.email, hashed_password=model.hashed_password)