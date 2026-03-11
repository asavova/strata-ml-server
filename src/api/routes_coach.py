"""Coaching endpoints."""

from fastapi import APIRouter

from .schemas import CoachRequest, CoachResponse

router = APIRouter(prefix="/coach", tags=["coaching"])

# Dependency injection — set by server.py at startup
_coach_engine = None


def set_coach(coach_engine):
    global _coach_engine
    _coach_engine = coach_engine


@router.post("/analyze", response_model=CoachResponse)
async def analyze(request: CoachRequest):
    """Analyze the position after a move was played."""
    last_move = request.lastMove.model_dump() if request.lastMove else None
    pieces = [p.model_dump() for p in request.pieces]

    result = _coach_engine.analyze(
        active_player=request.activePlayer,
        ply_count=request.plyCount,
        white_count=request.whitePieceCount or 0,
        black_count=request.blackPieceCount or 0,
        pieces=pieces,
        last_move=last_move,
    )
    return CoachResponse(**result)


@router.post("/suggest", response_model=CoachResponse)
async def suggest(request: CoachRequest):
    """Suggest strategic direction."""
    pieces = [p.model_dump() for p in request.pieces]

    result = _coach_engine.suggest(
        active_player=request.activePlayer,
        ply_count=request.plyCount,
        white_count=request.whitePieceCount or 0,
        black_count=request.blackPieceCount or 0,
        pieces=pieces,
    )
    return CoachResponse(**result)


@router.post("/teach", response_model=CoachResponse)
async def teach(request: CoachRequest):
    """Respond to a teaching request."""
    pieces = [p.model_dump() for p in request.pieces]

    result = _coach_engine.teach(
        active_player=request.activePlayer,
        ply_count=request.plyCount,
        white_count=request.whitePieceCount or 0,
        black_count=request.blackPieceCount or 0,
        pieces=pieces,
        question=request.playerQuestion,
    )
    return CoachResponse(**result)
