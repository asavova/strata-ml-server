# Future Task Opportunities

These opportunities are grounded in the current repository. They favor
deterministic behavior, multi-file impact, and clear test or Docker validation.

## Ranked Opportunities

1. Persist replay buffer snapshots across restarts.
   - Approval likelihood: high
   - Depth: training lifecycle, serialization, config paths, tests
   - Value: preserves self-play data and enables restart scenarios

2. Apply `TrainStepRequest` overrides safely.
   - Approval likelihood: high
   - Depth: API schema, trainer behavior, optimizer learning rates, tests
   - Value: makes an existing API contract meaningful

3. Add deterministic checkpoint metadata.
   - Approval likelihood: high
   - Depth: checkpoint format, load/save compatibility, health/status metadata
   - Value: improves model lifecycle inspection

4. Validate board coordinates and node IDs in coaching piece payloads.
   - Approval likelihood: high
   - Depth: schema validation, board constants, API tests
   - Value: prevents invalid game state from entering strategy logic

5. Implement REST/proto parity checks.
   - Approval likelihood: medium-high
   - Depth: protocol documentation, converter utilities, tests
   - Value: keeps Java/Python integration from drifting

6. Normalize and validate training policy targets.
   - Approval likelihood: medium-high
   - Depth: schema validation, tensor conversion, trainer loss behavior
   - Value: prevents malformed MCTS targets from corrupting training

7. Add stable replay-buffer sampling controls for tests.
   - Approval likelihood: medium-high
   - Depth: replay buffer API, trainer configuration, tests
   - Value: makes training tests reproducible without weakening production randomization

8. Track evaluation and coaching counters in `/health` or status endpoints.
   - Approval likelihood: medium
   - Depth: component state, route dependencies, tests
   - Value: exposes operational state without cloud infrastructure

9. Implement in-memory analysis history for coaching.
   - Approval likelihood: medium
   - Depth: request identity, history retention, API behavior, tests
   - Value: creates lifecycle/state-transition tasks beyond one-shot responses

10. Add Docker smoke tests and dependency-lock documentation.
    - Approval likelihood: medium
    - Depth: build reproducibility, packaging, docs
    - Value: improves evaluator confidence and future task setup

11. Add gRPC implementation for the existing proto services.
    - Approval likelihood: medium
    - Depth: service layer, generated code, Java compatibility, tests
    - Value: completes an advertised integration surface

12. Add checkpoint rollback when a training step produces non-finite loss.
    - Approval likelihood: medium
    - Depth: training lifecycle, model state management, tests
    - Value: hardens long-running self-play training

## Five Best SWE-bench-Style Candidates

1. Replay buffer persistence
   - Base commit: current behavior keeps examples only in memory.
   - Fix commit: add snapshot save/load plus status reporting and tests.

2. Train-step override application
   - Base commit: request fields are validated but ignored.
   - Fix commit: apply batch-size and learning-rate overrides for one step,
     preserving configured defaults afterward.

3. Coordinate and node validation
   - Base commit: coaching accepts impossible board coordinates.
   - Fix commit: reject invalid coordinates and inconsistent `nodeId` values.

4. Policy target normalization
   - Base commit: training accepts arbitrary policy vectors.
   - Fix commit: reject empty/negative policies and normalize valid visit-count
     targets before tensor conversion.

5. Checkpoint metadata and health reporting
   - Base commit: health reports a static model version.
   - Fix commit: persist checkpoint step/loss metadata and expose it through
     health/status endpoints.
