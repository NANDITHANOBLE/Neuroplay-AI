"""Tests for database ORM models — relationships and constraints."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from neuroplay.db.models import Base, Match, Move, User


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_user_match_relationship(db_session):
    user = User(username="test_user")
    db_session.add(user)
    db_session.commit()

    match = Match(user_id=user.id, mode="keyboard")
    db_session.add(match)
    db_session.commit()

    assert match.user.username == "test_user"
    assert user.matches[0].id == match.id


def test_move_requires_valid_result(db_session):
    user = User(username="test_user2")
    db_session.add(user)
    db_session.commit()
    match = Match(user_id=user.id, mode="keyboard")
    db_session.add(match)
    db_session.commit()

    move = Move(
        match_id=match.id,
        round_number=1,
        player_move=0,
        ai_move=1,
        result="win",
    )
    db_session.add(move)
    db_session.commit()
    assert move.result == "win"
