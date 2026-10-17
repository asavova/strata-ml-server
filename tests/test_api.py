"""Integration tests for the API endpoints."""

import inspect

import pytest
import torch
from httpx import AsyncClient, ASGITransport

from src.api import routes_coach, routes_eval, routes_train
from src.coach import CoachEngine
from src.evaluation import Evaluator
from src.network import StrataNetwork
from src.server import app
from src.training import ReplayBuffer


def test_coach_handlers_run_in_threadpool():
    """Synchronous coach work must not block the ASGI event loop."""
    assert not inspect.iscoroutinefunction(routes_coach.analyze)
    assert not inspect.iscoroutinefunction(routes_coach.suggest)
    assert not inspect.iscoroutinefunction(routes_coach.teach)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_evaluate(client, monkeypatch):
    network = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    monkeypatch.setattr(
        routes_eval, "_evaluator", Evaluator(network, torch.device("cpu"))
    )
    payload = {
        "state": [0.0] * 1008,
        "legal_mask": [1.0] * 15876,
    }
    response = await client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "value" in data
    assert "policy" in data
    assert -1.0 <= data["value"] <= 1.0


@pytest.mark.asyncio
async def test_submit_training(client, monkeypatch):
    monkeypatch.setattr(routes_train, "_replay_buffer", ReplayBuffer(max_size=100))
    payload = {
        "examples": [
            {
                "state": [0.5] * 1008,
                "policy": [0.0] * 15876,
                "value": 0.5,
            }
        ]
    }
    response = await client.post("/train/submit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_coach_analyze(client, monkeypatch):
    monkeypatch.setattr(routes_coach, "_coach_engine", CoachEngine())
    payload = {
        "requestType": "ANALYZE",
        "activePlayer": "WHITE",
        "plyCount": 5,
        "whitePieceCount": 25,
        "blackPieceCount": 27,
        "pieces": [],
    }
    response = await client.post("/coach/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "commentary" in data
