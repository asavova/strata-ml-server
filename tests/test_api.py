"""Integration tests for the API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from src.server import app
from src.api import routes_coach, routes_eval, routes_train

fake_replay_buffer = None


@pytest.fixture
async def client():
    global fake_replay_buffer
    fake_replay_buffer = FakeReplayBuffer()
    routes_eval.set_evaluator(FakeEvaluator())
    routes_train.set_dependencies(FakeTrainer(), fake_replay_buffer)
    routes_coach.set_coach(FakeCoach())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class FakeEvaluator:
    def __init__(self):
        self.eval_count = 0

    def evaluate(self, state, legal_mask):
        self.eval_count += 1
        return 0.25, [1.0 / len(legal_mask)] * len(legal_mask)

    def evaluate_batch(self, states, legal_masks):
        self.eval_count += len(states)
        return [(0.25, [1.0 / len(mask)] * len(mask)) for mask in legal_masks]

    def stats(self):
        return {"total_evaluations": self.eval_count}


class FakeReplayBuffer:
    def __init__(self):
        self.examples = []

    def add_batch(self, examples):
        self.examples.extend(examples)

    def size(self):
        return len(self.examples)

    def stats(self):
        return {"size": len(self.examples)}


class FakeTrainer:
    def train_step(self, replay_buffer, batch_size=None, learning_rate=None):
        return {"success": False, "reason": "Not enough data", "loss": 0.0}

    def get_status(self):
        return {
            "step_count": 0,
            "best_loss": 0.0,
            "recent_avg_loss": 0.0,
            "lr": 0.001,
            "latest_checkpoint": None,
        }


class FakeCoach:
    def stats(self):
        return {"total_requests": 0, "requests_by_type": {}, "llm_available": False}

    def analyze(self, **kwargs):
        return {"commentary": "Rule-based analysis.", "confidence": 0.75}

    def suggest(self, **kwargs):
        return {"commentary": "Rule-based suggestion.", "confidence": 0.75}

    def teach(self, **kwargs):
        return {"commentary": "Rule-based teaching.", "confidence": 0.75}


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["evaluation_stats"]["total_evaluations"] == 0
    assert data["coach_stats"]["total_requests"] == 0


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
    policy = [0.0] * 15876
    policy[10] = 3.0
    policy[20] = 1.0
    payload = {
        "examples": [
            {
                "state": [0.5] * 1008,
                "policy": policy,
                "value": 0.5,
            }
        ]
    }
    response = await client.post("/train/submit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert fake_replay_buffer.examples[0].policy[10] == 0.75
    assert fake_replay_buffer.examples[0].policy[20] == 0.25


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


@pytest.mark.asyncio
async def test_coach_rejects_invalid_piece_coordinates(client):
    payload = {
        "requestType": "ANALYZE",
        "activePlayer": "WHITE",
        "pieces": [
            {
                "nodeId": 0,
                "layer": 7,
                "row": 0,
                "col": 0,
                "owner": "WHITE",
            }
        ],
    }
    response = await client.post("/coach/analyze", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_coach_rejects_inconsistent_node_coordinates(client):
    payload = {
        "requestType": "ANALYZE",
        "activePlayer": "WHITE",
        "pieces": [
            {
                "nodeId": 5,
                "layer": 0,
                "row": 0,
                "col": 0,
                "owner": "WHITE",
            }
        ],
    }
    response = await client.post("/coach/analyze", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_evaluate_rejects_wrong_state_size(client):
    payload = {
        "state": [0.0] * 1007,
        "legal_mask": [1.0] * 15876,
    }
    response = await client.post("/evaluate", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_evaluate_rejects_empty_legal_mask(client):
    payload = {
        "state": [0.0] * 1008,
        "legal_mask": [0.0] * 15876,
    }
    response = await client.post("/evaluate", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_evaluate_rejects_mismatched_batch_lengths(client):
    payload = {
        "states": [[0.0] * 1008, [0.0] * 1008],
        "legal_masks": [[1.0] * 15876],
    }
    response = await client.post("/evaluate/batch", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_training_rejects_bad_value(client):
    payload = {
        "examples": [
            {
                "state": [0.5] * 1008,
                "policy": [0.0] * 15876,
                "value": 2.0,
            }
        ]
    }
    response = await client.post("/train/submit", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_training_rejects_empty_policy(client):
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
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_training_rejects_negative_policy(client):
    policy = [0.0] * 15876
    policy[0] = 1.0
    policy[1] = -0.5
    payload = {
        "examples": [
            {
                "state": [0.5] * 1008,
                "policy": policy,
                "value": 0.5,
            }
        ]
    }
    response = await client.post("/train/submit", json=payload)
    assert response.status_code == 422
