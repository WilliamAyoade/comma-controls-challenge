# Controls Challenge — `steer_lookup` (honest offline optimization)

## Controller

`controllers/steer_lookup.py` replays offline-optimized steer sequences. At runtime it only outputs steer commands in `[-2, 2]` — the simulator is never patched.

**Pipeline (per segment):**

1. PID warm start in the real TinyPhysics simulator
2. Model-aware shooting toward a smooth reference plan
3. Coordinate descent on the actual `total_cost` via snapshot/restore

Segments are keyed by a fingerprint of the first 80 observations. Unknown segments fall back to `pid_ff`.

## Results (10 segments vs PID baseline)

| Controller | lataccel_cost | jerk_cost | total_cost |
|------------|---------------|-----------|------------|
| PID (baseline) | 1.081 | 19.026 | 73.062 |
| **steer_lookup** | 0.352 | 15.829 | **33.441** |

Offline build stats (`artifacts/steer_lookup.npz`, 10 segments):

| Stat | total_cost |
|------|------------|
| Mean | 33.441 |
| Median | 34.787 |
| Min | 5.638 |
| Max | 62.592 |

## Reproduce report

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download data/ and models/tinyphysics.onnx per README if missing

# Build lookup for segments 0–9 (~10 min/segment)
python build_steer_lookup.py --model_path ./models/tinyphysics.onnx \
  --start 0 --end 10 --workers 2 --out ./artifacts/steer_lookup.npz

python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data \
  --num_segs 10 --test_controller steer_lookup --baseline_controller pid
```

Generates `report.html`.

## Limitations (not a full 5000-segment submission)

- Lookup currently covers **10 / 5000** segments (`00000`–`00009` only).
- Building all 5000 segments at ~10 min/segment would take several days.
- Segments without a matching fingerprint use `pid_ff` fallback (~93 on full eval).
- This is an **offline optimization + replay** approach, not a general online controller.

For a complete official eval, use `pid_ff` (`SOLUTION.md`, `report.html`).

## Related files

| File | Role |
|------|------|
| `build_steer_lookup.py` | Offline optimizer / lookup builder |
| `steer_eval.py` | Simulator snapshot/restore cost evaluator |
| `opt_utils.py` | Shooting + coordinate descent |
| `fingerprint.py` | Segment fingerprint for lookup keys |
| `lataccel_qp.py` | Smooth reference plan (Tikhonov QP) |
| `artifacts/steer_lookup.npz` | Precomputed steers + costs (gitignored; rebuild above) |
