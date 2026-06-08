# Strata ML Server API

The server exposes a JSON REST API on port 9000. The Java Strata engine is the
primary caller, but the endpoints are deterministic and can be tested directly.

## Health

`GET /health`

Returns service status, selected device, and model version metadata.

## Evaluation

`POST /evaluate`

Request:

```json
{
  "state": [0.0],
  "legal_mask": [1.0]
}
```

Contract:

- `state` must contain exactly 1008 floats.
- `legal_mask` must contain exactly 15876 floats.
- `legal_mask` must contain at least one positive value.

Response:

```json
{
  "value": 0.12,
  "policy": [0.0]
}
```

`POST /evaluate/batch`

Request:

```json
{
  "states": [[0.0]],
  "legal_masks": [[1.0]]
}
```

Contract:

- `states` and `legal_masks` must both be non-empty.
- Each state and mask follows the single-evaluation shape rules.
- The number of states must match the number of masks.

## Training

`POST /train/submit`

Adds self-play examples to the replay buffer and writes the configured replay
snapshot.

Request:

```json
{
  "examples": [
    {
      "state": [0.5],
      "policy": [0.0],
      "value": 0.5
    }
  ]
}
```

Contract:

- `examples` must be non-empty.
- Each `state` must contain exactly 1008 floats.
- Each `policy` must contain exactly 15876 floats.
- `value` must be between -1.0 and 1.0.

`POST /train/step`

Runs one trainer step against the current replay buffer. Optional request fields:

```json
{
  "batch_size": 32,
  "learning_rate": 0.001
}
```

Both optional fields must be positive when provided. The current trainer uses
configured training values when these fields are omitted. When provided,
`batch_size` controls the sample size for that step and `learning_rate` is used
only for that optimizer step.

`GET /train/status`

Returns step count, current learning rate, best loss, recent average loss, and
replay buffer statistics.

## Coaching

`POST /coach/analyze`

Analyzes the latest move and current position.

`POST /coach/suggest`

Returns a strategic direction for the active player.

`POST /coach/teach`

Answers a player question using deterministic rule-based coaching by default.

Piece payload contract:

- `nodeId` must be between 0 and 125.
- `layer` must be between 0 and 6.
- `row` must be between 0 and 2.
- `col` must be between 0 and 5.
- When `nodeId`, `layer`, `row`, and `col` are all supplied, `nodeId` must equal
  `layer * 18 + row * 6 + col`.

The optional LLM integration is disabled in `config/default.yaml`. When enabled
with a model name and `STRATA_LLM_API_KEY`, the LLM is a replaceable commentary
provider; rule-based coaching remains the fallback.
