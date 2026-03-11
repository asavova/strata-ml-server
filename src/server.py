"""
Strata ML Server — main application entry point.

Wires together: network, trainer, evaluator, coach, API routes.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI

from .config import load_config
from .utils.logging_config import setup_logging
from .network import StrataNetwork
from .training import Trainer, ReplayBuffer
from .evaluation import Evaluator
from .coach import CoachEngine
from .api import routes_eval, routes_train, routes_coach


# ── Application state (initialized in lifespan) ───────────

config = load_config()
logger = setup_logging(config.paths.log_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("=" * 60)
    logger.info("  STRATA ML SERVER starting")
    logger.info("=" * 60)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create directories
    for d in [config.paths.model_dir, config.paths.data_dir, config.paths.log_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Initialize neural network
    network = StrataNetwork(
        state_size=config.model.state_size,
        hidden_size=config.model.hidden_size,
        policy_size=config.model.policy_size,
        num_residual_blocks=config.model.num_residual_blocks,
        dropout=config.model.dropout,
    ).to(device)

    # Load existing weights if available
    network.load(config.paths.latest_model, device)

    # Initialize components
    replay_buffer = ReplayBuffer(max_size=config.training.max_replay_buffer)
    trainer = Trainer(network, device, config.training, config.paths)
    evaluator = Evaluator(network, device)
    coach_engine = CoachEngine(
        llm_model=config.coach.llm_model,
        llm_api_key_env=config.coach.llm_api_key_env,
        llm_base_url=config.coach.llm_base_url,
    )

    # Inject dependencies into route modules
    routes_eval.set_evaluator(evaluator)
    routes_train.set_dependencies(trainer, replay_buffer)
    routes_coach.set_coach(coach_engine)

    logger.info("All components initialized. Server is ready.")

    yield  # Server runs here

    # Shutdown
    logger.info("Saving final model checkpoint...")
    network.save(config.paths.latest_model)
    logger.info("Server shut down cleanly.")


# ── FastAPI application ───────────────────────────────────

app = FastAPI(
    title="Strata ML Server",
    version="1.0.0",
    description="PyTorch neural network server for the Strata 3D board-game engine",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint used by Java NeuralNetClient.connect()."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "ok",
        "service": "strata-ml-server",
        "device": device,
        "model_version": "1.0",
    }


# Register route modules
app.include_router(routes_eval.router)
app.include_router(routes_train.router)
app.include_router(routes_coach.router)


# ── CLI entry point ───────────────────────────────────────

def main():
    uvicorn.run(
        "src.server:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
