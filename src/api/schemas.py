"""
Pydantic schemas for the REST API.
Shared between all route modules.
"""

from pydantic import BaseModel
from typing import Optional


# ── Evaluation ─────────────────────────────────────────────

class EvalRequest(BaseModel):
    state: list[float]
    legal_mask: list[float]


class EvalResponse(BaseModel):
    value: float
    policy: list[float]


class BatchEvalRequest(BaseModel):
    states: list[list[float]]
    legal_masks: list[list[float]]


class BatchEvalResponse(BaseModel):
    results: list[EvalResponse]


# ── Training ──────────────────────────────────────────────

class TrainingExampleSchema(BaseModel):
    state: list[float]
    policy: list[float]
    value: float


class TrainingBatchRequest(BaseModel):
    examples: list[TrainingExampleSchema]


class TrainStepRequest(BaseModel):
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None


class TrainResponse(BaseModel):
    success: bool
    loss: float = 0.0
    examples_processed: int = 0
    step: int = 0
    message: str = ""


class TrainStatusResponse(BaseModel):
    step_count: int
    best_loss: float
    recent_avg_loss: float
    lr: float
    buffer_size: int
    buffer_stats: dict


# ── Coaching ──────────────────────────────────────────────

class CoordSchema(BaseModel):
    layer: int = 0
    row: int = 0
    col: int = 0


class LastMoveSchema(BaseModel):
    movedPiece: Optional[str] = None
    mover: Optional[str] = None
    isCapture: bool = False
    capturedPiece: Optional[str] = None
    newStage: Optional[str] = None

    class Config:
        extra = "allow"


class PieceSchema(BaseModel):
    nodeId: int = 0
    layer: int = 0
    row: int = 0
    col: int = 0
    owner: str = "WHITE"
    type: str = "ADEPT"
    stage: str = "CALCINATION"

    class Config:
        extra = "allow"


class CoachRequest(BaseModel):
    requestType: str = "ANALYZE"
    activePlayer: str = "WHITE"
    plyCount: int = 0
    playerQuestion: Optional[str] = None
    lastMove: Optional[LastMoveSchema] = None
    whitePieceCount: Optional[int] = 0
    blackPieceCount: Optional[int] = 0
    pieces: list[PieceSchema] = []

    class Config:
        extra = "allow"


class CoachResponse(BaseModel):
    commentary: str
    suggestion: str = ""
    teachingTip: str = ""
    confidence: float = 0.75
