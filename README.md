# Strata ML Server

PyTorch neural network server for the Strata 3D board-game engine.

## Architecture

```
Java Engine (strata-engine)
│
│ HTTP / gRPC on localhost:9000
▼
Python ML Server (this project)
├── Neural network (policy + value heads)
├── Self-play training pipeline
├── Replay buffer
└── LLM coaching integration
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
python -m src.server
```

Server starts on http://localhost:9000.

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server status + model info |
| `/evaluate` | POST | Evaluate a board position |
| `/evaluate/batch` | POST | Batch position evaluation |
| `/train/submit` | POST | Submit self-play training data |
| `/train/step` | POST | Trigger a training step |
| `/train/status` | GET | Training pipeline status |
| `/coach/analyze` | POST | Position analysis commentary |
| `/coach/suggest` | POST | Move suggestion |
| `/coach/teach` | POST | Strategic teaching |

## Training Loop

1. Java engine runs self-play games
2. Training examples are POSTed to `/train/submit`
3. Server accumulates in replay buffer
4. `/train/step` runs a gradient update
5. Updated model is used for next evaluation cycle

---

## `config/default.yaml`

```yaml
server:
  host: "0.0.0.0"
  port: 9000

model:
  state_size: 1008          # 126 nodes * 8 channels
  policy_size: 15876         # 126 * 126 possible moves
  hidden_size: 512
  num_residual_blocks: 6
  dropout: 0.1

training:
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 256
  max_replay_buffer: 200000
  min_buffer_for_training: 128
  checkpoint_interval: 50    # save every N training steps
  lr_schedule:
    step_milestones: [1000, 5000, 10000]
    gamma: 0.5

paths:
  model_dir: "models"
  data_dir: "data"
  log_dir: "logs"
  latest_model: "models/latest.pt"
  best_model: "models/best.pt"

coach:
  # Set to null to use rule-based coaching only
  # Set to a model name to use LLM (e.g., "gpt-4o-mini")
  llm_model: null
  # Your API key — set via STRATA_LLM_API_KEY env var, never hardcode
  llm_api_key_env: "STRATA_LLM_API_KEY"
  llm_base_url: "https://api.openai.com/v1"
```

## How to run

```bash
# Setup
cd strata-ml-server
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements.txt

# Run the server
python -m src

# Run tests
pip install pytest pytest-asyncio
pytest tests/ -v

# Java engine connects automatically to http://localhost:9000
```

## LLM coaching — how to enable

Set an environment variable with your API key and update `config/default.yaml`:

```bash
# Windows
set STRATA_LLM_API_KEY=your-key-here

# Linux/Mac
export STRATA_LLM_API_KEY=your-key-here
```

Without the key, the coach runs in **rule-based mode** — still functional, just less conversational.

## Connection summary

| Java class | Python endpoint | Purpose |
|---|---|---|
| `NeuralNetClient.connect()` | `GET /health` | Startup handshake |
| `NeuralNetClient.evaluate()` | `POST /evaluate` | Position evaluation |
| `NeuralNetClient.submitTrainingData()` | `POST /train/submit` | Self-play data |
| `CoachClient.analyze()` | `POST /coach/analyze` | Post-move commentary |
| `CoachClient.suggest()` | `POST /coach/suggest` | Move suggestion |
| `CoachClient.teach()` | `POST /coach/teach` | Strategic Q&A |

Both repos share the same protocol via `strata.proto` and the JSON schemas. The Java engine drives the game and generates training data. The Python server trains the neural network and provides coaching.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
