"""Integration tests for the API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from src.server import app


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
async def test_evaluate(client):
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
async def test_submit_training(client):
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
async def test_coach_analyze(client):
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
