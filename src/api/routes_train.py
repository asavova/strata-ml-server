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
_replay_snapshot_path = None


def set_dependencies(trainer, replay_buffer, replay_snapshot_path=None):
    global _trainer, _replay_buffer, _replay_snapshot_path
    _trainer = trainer
    _replay_buffer = replay_buffer
    _replay_snapshot_path = replay_snapshot_path


@router.post("/submit", response_model=TrainResponse)
async def submit_training_data(batch: TrainingBatchRequest):
    """Receive self-play training data from the Java engine."""
    examples = [
        TrainingExample(state=ex.state, policy=ex.policy, value=ex.value)
        for ex in batch.examples
    ]
    _replay_buffer.add_batch(examples)
    if _replay_snapshot_path:
        _replay_buffer.save_snapshot(_replay_snapshot_path)

    return TrainResponse(
        success=True,
        loss=0.0,
        examples_processed=len(examples),
        message=f"Added {len(examples)} examples. Buffer: {_replay_buffer.size()}"
    )


@router.post("/step", response_model=TrainResponse)
async def train_step(request: TrainStepRequest = None):
    """Trigger one training step."""
    request = request or TrainStepRequest()
    result = _trainer.train_step(
        _replay_buffer,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
    )
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
        latest_checkpoint=trainer_status.get("latest_checkpoint"),
    )
