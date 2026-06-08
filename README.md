# Strata ML Server

PyTorch neural-network and coaching server for the Strata 3D board-game engine.

The project is a backend service for deterministic game workflows: position
evaluation, self-play replay buffering, training, checkpointing, and rule-based
strategic coaching. Optional LLM commentary can be enabled, but the default
runtime works without an external AI provider.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m src
```

Server starts on `http://localhost:9000`.

## Docker

```bash
docker build -t strata-ml-server .
docker run --rm -p 9000:9000 strata-ml-server
docker run --rm strata-ml-server python -m pytest -q
```

## Main Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server status and model metadata |
| `/evaluate` | POST | Evaluate one board position |
| `/evaluate/batch` | POST | Evaluate multiple positions |
| `/train/submit` | POST | Submit self-play training data |
| `/train/step` | POST | Trigger one training step |
| `/train/status` | GET | Training and replay-buffer status |
| `/coach/analyze` | POST | Analyze a recent move |
| `/coach/suggest` | POST | Suggest strategic direction |
| `/coach/teach` | POST | Answer a strategy question |

See [API.md](API.md) for request and validation details.

## Architecture

```text
Java Strata Engine
        |
        | REST JSON on localhost:9000
        v
Python ML Server
  - FastAPI route layer
  - fixed-shape schema validation
  - policy/value neural network
  - replay buffer and trainer
  - deterministic rule-based coaching
  - optional LLM commentary fallback
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for component details and current gaps.

## Training Loop

1. The Java engine runs self-play games.
2. Training examples are posted to `/train/submit`.
3. The server stores examples in a bounded replay buffer.
4. `/train/step` samples a mini-batch and runs a gradient update.
5. Checkpoints are written to `models/latest.pt` and `models/best.pt`.

## Configuration

Defaults live in `config/default.yaml`.

Important values:

```yaml
server:
  host: "0.0.0.0"
  port: 9000

model:
  state_size: 1008
  policy_size: 15876

training:
  batch_size: 256
  min_buffer_for_training: 128
  checkpoint_interval: 50

coach:
  llm_model: null
  llm_api_key_env: "STRATA_LLM_API_KEY"
```

With `llm_model: null`, coaching uses deterministic rule-based responses. To
enable optional LLM commentary, set a model name and provide
`STRATA_LLM_API_KEY` in the environment.

## Tests

```bash
python -m pytest -q
```

The test suite covers network shape behavior, legal-mask handling, evaluator
batching, replay-buffer/trainer behavior, API validation, and deterministic
coaching rules.

## Future Task Planning

See [TASK_OPPORTUNITIES.md](TASK_OPPORTUNITIES.md) for ranked future
SWE-bench-style task candidates.
