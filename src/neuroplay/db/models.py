"""
SQLAlchemy ORM models for NeuroPlay-AI.
Mirrors the schema in database/schema/*.sql — this is the source of truth
used by the application; the raw SQL files are for documentation/manual setup.
"""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_matches = Column(Integer, default=0)
    total_wins = Column(Integer, default=0)

    matches = relationship("Match", back_populates="user")
    psychology_profile = relationship("PsychologyProfile", back_populates="user", uselist=False)


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    mode = Column(String(20), nullable=False, default="keyboard")
    final_score_user = Column(Integer, default=0)
    final_score_ai = Column(Integer, default=0)
    model_used = Column(String(30), default="markov")

    user = relationship("User", back_populates="matches")
    moves = relationship("Move", back_populates="match")
    drift_events = relationship("DriftEvent", back_populates="match")


class Move(Base):
    __tablename__ = "moves"
    __table_args__ = (
        CheckConstraint("player_move IN (0,1,2)"),
        CheckConstraint("ai_move IN (0,1,2)"),
        CheckConstraint("result IN ('win','loss','draw')"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    player_move = Column(Integer, nullable=False)
    ai_move = Column(Integer, nullable=False)
    result = Column(String(10), nullable=False)
    reaction_time_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="moves")
    predictions = relationship("Prediction", back_populates="move")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    move_id = Column(Integer, ForeignKey("moves.id"), nullable=False)
    model_name = Column(String(30), nullable=False)
    predicted_move = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    explanation_json = Column(Text, nullable=True)

    move = relationship("Move", back_populates="predictions")


class DriftEvent(Base):
    __tablename__ = "drift_events"
    __table_args__ = (CheckConstraint("severity IN ('warning','drift')"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    detected_at_round = Column(Integer, nullable=False)
    detector_method = Column(String(20), nullable=False)
    severity = Column(String(10), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="drift_events")


class PsychologyProfile(Base):
    __tablename__ = "psychology_profiles"
    __table_args__ = (UniqueConstraint("user_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dominant_pattern = Column(String(30), nullable=True)
    lz_complexity_score = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="psychology_profile")
