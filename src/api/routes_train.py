"""Training endpoints."""

from fastapi import APIRouter

from .schemas import (
    TrainingBatchRequest,
    TrainStepRequest,
    TrainResponse,
    TrainStatusResponse,
)
from ..training.replay_buffer import TrainingExample

router = APIRouter(prefix="/train", tags=["training"])

# Dependency injection — set by server.py at startup
_trainer = None
_replay_buffer = None


def set_dependencies(trainer, replay_buffer):
    global _trainer, _replay_buffer
    _trainer = trainer
    _replay_buffer = replay_buffer


@router.post("/submit", response_model=TrainResponse)
async def submit_training_data(batch: TrainingBatchRequest):
    """Receive self-play training data from the Java engine."""
    examples = [
        TrainingExample(state=ex.state, policy=ex.policy, value=ex.value)
        for ex in batch.examples
    ]
    _replay_buffer.add_batch(examples)

    return TrainResponse(
        success=True,
        loss=0.0,
        examples_processed=len(examples),
        message=f"Added {len(examples)} examples. Buffer: {_replay_buffer.size()}"
    )


@router.post("/step", response_model=TrainResponse)
async def train_step(request: TrainStepRequest = None):
    """Trigger one training step."""
    result = _trainer.train_step(_replay_buffer)
    return TrainResponse(
        success=result.get("success", False),
        loss=result.get("loss", 0.0),
        examples_processed=result.get("buffer_size", 0),
        step=result.get("step", 0),
        message=result.get("reason", "OK"),
    )


@router.get("/status", response_model=TrainStatusResponse)
async def training_status():
    """Get training pipeline status."""
    trainer_status = _trainer.get_status()
    buffer_stats = _replay_buffer.stats()
    return TrainStatusResponse(
        step_count=trainer_status["step_count"],
        best_loss=trainer_status["best_loss"],
        recent_avg_loss=trainer_status["recent_avg_loss"],
        lr=trainer_status["lr"],
        buffer_size=_replay_buffer.size(),
        buffer_stats=buffer_stats,
    )
