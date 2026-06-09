"""Evaluation endpoints."""

from fastapi import APIRouter, Depends

from .schemas import EvalRequest, EvalResponse, BatchEvalRequest, BatchEvalResponse

router = APIRouter(prefix="/evaluate", tags=["evaluation"])

# Dependency injection set by server.py at startup
_evaluator = None


def set_evaluator(evaluator):
    global _evaluator
    _evaluator = evaluator


def get_stats() -> dict:
    if _evaluator is None:
        return {"total_evaluations": 0}
    return _evaluator.stats()


@router.post("", response_model=EvalResponse)
async def evaluate_position(request: EvalRequest):
    """Evaluate a single board position."""
    value, policy = _evaluator.evaluate(request.state, request.legal_mask)
    return EvalResponse(value=value, policy=policy)


@router.post("/batch", response_model=BatchEvalResponse)
async def evaluate_batch(request: BatchEvalRequest):
    """Evaluate multiple positions in one forward pass."""
    results = _evaluator.evaluate_batch(request.states, request.legal_masks)
    return BatchEvalResponse(
        results=[EvalResponse(value=v, policy=p) for v, p in results]
    )
