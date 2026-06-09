"""Tests for rule-based coaching behavior."""

from src.coach.coach_engine import CoachEngine


def test_rule_analyze_is_repeatable_for_same_position():
    coach = CoachEngine(llm_model=None)
    pieces = [
        {"layer": 3, "owner": "WHITE", "stage": "CALCINATION"},
        {"layer": 3, "owner": "BLACK", "stage": "ILLUMINATION"},
    ]
    last_move = {
        "isCapture": True,
        "mover": "WHITE",
        "movedPiece": "ELEMENTAL",
        "capturedPiece": "ADEPT",
        "newStage": "DISSOLUTION",
    }

    first = coach.analyze("WHITE", 6, 24, 23, pieces, last_move)
    second = coach.analyze("WHITE", 6, 24, 23, pieces, last_move)

    assert first == second
    assert "Center control" in first["commentary"]
    assert first["confidence"] >= 0.65
    assert first["confidence"] <= 0.90


def test_rule_teach_answers_transformation_questions():
    coach = CoachEngine(llm_model=None)

    response = coach.teach(
        active_player="BLACK",
        ply_count=18,
        white_count=20,
        black_count=21,
        pieces=[],
        question="How should I think about transformation stage timing?",
    )

    assert "advances through 9 transformation stages" in response["commentary"]
    assert response["confidence"] == 0.80


def test_rule_suggest_uses_center_presence():
    coach = CoachEngine(llm_model=None)

    response = coach.suggest(
        active_player="WHITE",
        ply_count=12,
        white_count=22,
        black_count=22,
        pieces=[{"layer": 3, "owner": "WHITE"} for _ in range(3)],
    )

    assert "Strong center presence" in response["suggestion"]


def test_coach_stats_track_request_types():
    coach = CoachEngine(llm_model=None)

    coach.analyze("WHITE", 1, 27, 27, [], None)
    coach.suggest("WHITE", 2, 27, 27, [])
    coach.teach("BLACK", 3, 26, 27, [], "center")

    stats = coach.stats()
    assert stats["total_requests"] == 3
    assert stats["requests_by_type"]["analyze"] == 1
    assert stats["requests_by_type"]["suggest"] == 1
    assert stats["requests_by_type"]["teach"] == 1
    assert stats["llm_available"] is False
