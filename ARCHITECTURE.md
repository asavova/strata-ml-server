# Architecture

Strata ML Server is a backend companion for the Strata 3D board-game engine. It
evaluates encoded board states, stores self-play examples, trains a dual-headed
network, and produces strategic coaching. The LLM integration is optional and
replaceable; core behavior comes from deterministic board, training, and rule
workflows.

## Runtime Components

- `src/server.py` wires configuration, logging, model initialization, replay
  buffer, trainer, evaluator, and coaching routes.
- `src/api` defines JSON schemas and route handlers for evaluation, training,
  and coaching.
- `src/network` contains the PyTorch residual network with policy and value
  heads.
- `src/evaluation` converts API payloads into tensors and runs inference.
- `src/training` owns replay-buffer storage, tensor conversion, optimization,
  learning-rate scheduling, and checkpointing.
- `src/coach` owns deterministic rule-based coaching and optional LLM fallback.
- `src/utils/proto_converter.py` captures board geometry constants shared with
  the Java engine.
- `proto/strata.proto` documents the intended gRPC protocol shared with the Java
  engine.

## Data Flow

1. The Java engine encodes a Strata board into a 1008-float state vector and a
   15876-float legal move mask.
2. `/evaluate` or `/evaluate/batch` validates payload shape and calls the
   evaluator.
3. The evaluator runs the network and returns a scalar value plus move policy.
4. Self-play examples are posted to `/train/submit` and retained in a bounded
   replay buffer.
5. `/train/step` samples from the replay buffer, updates model weights, advances
   the scheduler, and checkpoints at configured intervals.
6. Coaching endpoints inspect position metadata and return deterministic
   strategic text unless an LLM provider is explicitly configured.

## Deterministic Contracts

- Evaluation states are fixed at 126 nodes times 8 channels.
- Policy and legal-mask vectors are fixed at 126 by 126 possible moves.
- Empty legal masks are rejected before network inference to avoid undefined
  softmax behavior.
- Training values are constrained to the game outcome range `[-1.0, 1.0]`.
- Rule-based coaching uses stable request-keyed selection rather than process
  randomness.

## Persistence And Lifecycle

- Replay buffer entries live in memory and are capped by
  `training.max_replay_buffer`.
- Model checkpoints are written to `paths.latest_model`.
- The best observed checkpoint is written to `paths.best_model` when checkpoint
  loss improves.
- Runtime directories are created during FastAPI lifespan startup.

## Docker And Evaluation

The Dockerfile builds a Python 3.11 environment, installs the package with dev
test dependencies, and starts the FastAPI app with `python -m src`. Local model
checkpoints, replay data, logs, caches, and virtual environments are excluded
from the build context by `.dockerignore`.

Useful commands:

```bash
docker build -t strata-ml-server .
docker run --rm -p 9000:9000 strata-ml-server
docker run --rm strata-ml-server python -m pytest -q
```

## Current Gaps

- Replay buffer persistence is in-memory only.
- gRPC service definitions are present, but only REST routes are implemented.
- Dependency versions are lower-bounded rather than fully locked.
- Test execution requires installing dev dependencies.
