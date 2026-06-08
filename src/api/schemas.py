"""
Pydantic schemas for the REST API.
Shared between all route modules.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


STATE_SIZE = 1008
POLICY_SIZE = 15876
TOTAL_NODES = 126
NODES_PER_LAYER = 18
LAYERS = 7
ROWS = 3
COLUMNS = 6


def _ensure_length(values: list, expected: int, field_name: str) -> list:
    if len(values) != expected:
        raise ValueError(f"{field_name} must contain exactly {expected} values")
    return values


def _ensure_legal_mask(values: list[float], field_name: str) -> list[float]:
    _ensure_length(values, POLICY_SIZE, field_name)
    if not any(value > 0 for value in values):
        raise ValueError(f"{field_name} must contain at least one legal move")
    return values


class EvalRequest(BaseModel):
    state: list[float]
    legal_mask: list[float]

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: list[float]) -> list[float]:
        return _ensure_length(value, STATE_SIZE, "state")

    @field_validator("legal_mask")
    @classmethod
    def validate_legal_mask(cls, value: list[float]) -> list[float]:
        return _ensure_legal_mask(value, "legal_mask")


class EvalResponse(BaseModel):
    value: float
    policy: list[float]


class BatchEvalRequest(BaseModel):
    states: list[list[float]]
    legal_masks: list[list[float]]

    @field_validator("states")
    @classmethod
    def validate_states(cls, value: list[list[float]]) -> list[list[float]]:
        if not value:
            raise ValueError("states must contain at least one position")
        for state in value:
            _ensure_length(state, STATE_SIZE, "state")
        return value

    @field_validator("legal_masks")
    @classmethod
    def validate_legal_masks(cls, value: list[list[float]]) -> list[list[float]]:
        if not value:
            raise ValueError("legal_masks must contain at least one mask")
        for mask in value:
            _ensure_legal_mask(mask, "legal_mask")
        return value

    @model_validator(mode="after")
    def validate_batch_lengths(self):
        if len(self.states) != len(self.legal_masks):
            raise ValueError("states and legal_masks must contain the same number of items")
        return self


class BatchEvalResponse(BaseModel):
    results: list[EvalResponse]


class TrainingExampleSchema(BaseModel):
    state: list[float]
    policy: list[float]
    value: float = Field(ge=-1.0, le=1.0)

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: list[float]) -> list[float]:
        return _ensure_length(value, STATE_SIZE, "state")

    @field_validator("policy")
    @classmethod
    def validate_policy(cls, value: list[float]) -> list[float]:
        return _ensure_length(value, POLICY_SIZE, "policy")


class TrainingBatchRequest(BaseModel):
    examples: list[TrainingExampleSchema]

    @field_validator("examples")
    @classmethod
    def validate_examples(cls, value: list[TrainingExampleSchema]) -> list[TrainingExampleSchema]:
        if not value:
            raise ValueError("examples must contain at least one training example")
        return value


class TrainStepRequest(BaseModel):
    batch_size: Optional[int] = Field(default=None, gt=0)
    learning_rate: Optional[float] = Field(default=None, gt=0)


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


class CoordSchema(BaseModel):
    layer: int = Field(default=0, ge=0, lt=LAYERS)
    row: int = Field(default=0, ge=0, lt=ROWS)
    col: int = Field(default=0, ge=0, lt=COLUMNS)


class LastMoveSchema(BaseModel):
    movedPiece: Optional[str] = None
    mover: Optional[str] = None
    isCapture: bool = False
    capturedPiece: Optional[str] = None
    newStage: Optional[str] = None

    class Config:
        extra = "allow"


class PieceSchema(BaseModel):
    nodeId: int = Field(default=0, ge=0, lt=TOTAL_NODES)
    layer: int = Field(default=0, ge=0, lt=LAYERS)
    row: int = Field(default=0, ge=0, lt=ROWS)
    col: int = Field(default=0, ge=0, lt=COLUMNS)
    owner: str = "WHITE"
    type: str = "ADEPT"
    stage: str = "CALCINATION"

    @model_validator(mode="after")
    def validate_node_coordinates(self):
        supplied = self.model_fields_set
        if {"nodeId", "layer", "row", "col"}.issubset(supplied):
            expected = self.layer * NODES_PER_LAYER + self.row * COLUMNS + self.col
            if self.nodeId != expected:
                raise ValueError("nodeId must match layer, row, and col")
        return self

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
