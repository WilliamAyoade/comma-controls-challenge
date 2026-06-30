# Controls Challenge Submission — `pid_ff`

## Controller

`controllers/pid_ff.py` — PID with feedforward, 1s preview, and roll compensation.

## Results (5000 segments vs PID baseline)

| Controller | lataccel_cost | jerk_cost | total_cost |
|------------|---------------|-----------|------------|
| PID (baseline) | 1.695 | 25.490 | 110.255 |
| **pid_ff** | 1.300 | 28.361 | **93.363** |

## Reproduce

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download data/ and models/tinyphysics.onnx per README if missing

python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data \
  --num_segs 5000 --test_controller pid_ff --baseline_controller pid
```

Generates `report.html`.

## Other experiments (not submitted)

- `controllers/steer_lookup.py` — honest offline steer optimization (~33 on 10 segs); see `SOLUTION_STEER_LOOKUP.md` and `report.html`
- `controllers/continuous_lookup.py` — analytic metric floor (~6.9; injects lataccel, not real steering)
