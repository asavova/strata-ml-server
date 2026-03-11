"""
Configuration management.
Loads from YAML file and allows environment variable overrides.
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel
from typing import Optional


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9000


class ModelConfig(BaseModel):
    state_size: int = 1008
    policy_size: int = 15876
    hidden_size: int = 512
    num_residual_blocks: int = 6
    dropout: float = 0.1


class LRScheduleConfig(BaseModel):
    step_milestones: list[int] = [1000, 5000, 10000]
    gamma: float = 0.5


class TrainingConfig(BaseModel):
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 256
    max_replay_buffer: int = 200000
    min_buffer_for_training: int = 128
    checkpoint_interval: int = 50
    lr_schedule: LRScheduleConfig = LRScheduleConfig()


class PathsConfig(BaseModel):
    model_dir: str = "models"
    data_dir: str = "data"
    log_dir: str = "logs"
    latest_model: str = "models/latest.pt"
    best_model: str = "models/best.pt"


class CoachConfig(BaseModel):
    llm_model: Optional[str] = None
    llm_api_key_env: str = "STRATA_LLM_API_KEY"
    llm_base_url: str = "https://api.openai.com/v1"


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    paths: PathsConfig = PathsConfig()
    coach: CoachConfig = CoachConfig()


def load_config(config_path: str = "config/default.yaml") -> AppConfig:
    """Load configuration from YAML file, falling back to defaults."""
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig(**raw)
    return AppConfig()
