"""Honest per-segment steer optimization through the real simulator."""
import numpy as np
import pandas as pd

from fingerprint import fingerprint_from_csv
from lataccel_qp import solve_lataccel_qp
from steer_eval import SteerEvaluator
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE, TinyPhysicsSimulator
from controllers.pid_ff import Controller

# Coarse-to-fine schedule (RyanL2 steer_cd); 2 passes balances quality vs build time.
CD_SCHEDULE = (
  (0.3, 0.15, 0.07),
  (0.12, 0.06, 0.03),
  (0.05, 0.02, 0.01),
)


def pid_warmstart(model, data_path: str) -> np.ndarray:
  sim = TinyPhysicsSimulator(model, data_path, controller=Controller(), debug=False)
  sim.rollout()
  return np.array(sim.action_history)[CONTROL_START_IDX:COST_END_IDX].astype(np.float64)


def shoot_steer_toward_plan(ev: SteerEvaluator, plan: np.ndarray) -> None:
  """One-step model-aware steer pick toward a smooth reference plan."""
  grid = np.linspace(STEER_RANGE[0], STEER_RANGE[1], 15)
  ev.reset()
  ev.advance_to(CONTROL_START_IDX)
  for t in range(CONTROL_START_IDX, COST_END_IDX):
    desired = float(plan[t - CONTROL_START_IDX])
    best_s, best_err = float(ev.steer[t - CONTROL_START_IDX]), float('inf')
    for lat, steer in ev.reachable(grid):
      err = (lat - desired) ** 2
      if err < best_err:
        best_err, best_s = err, steer
    ev.set_steer_at(t, best_s)
    ev.step()


def coordinate_descent(ev: SteerEvaluator, schedule=CD_SCHEDULE, stop_at: float = 40.0) -> float:
  """Per-timestep CD on full realized simulator cost (never worse than warm start)."""
  for deltas in schedule:
    ev.reset()
    ev.advance_to(CONTROL_START_IDX)
    for t in range(CONTROL_START_IDX, COST_END_IDX):
      snap = ev.snapshot()
      ev.restore(snap)
      cur_cost = ev.roll_cost()['total_cost']
      best_s = float(ev.steer[t - CONTROL_START_IDX])
      for d in deltas:
        for cand in (best_s + d, best_s - d):
          cand = float(np.clip(cand, STEER_RANGE[0], STEER_RANGE[1]))
          if cand == best_s:
            continue
          ev.set_steer_at(t, cand)
          ev.restore(snap)
          c = ev.roll_cost()['total_cost']
          if c < cur_cost - 1e-9:
            cur_cost = c
            best_s = cand
      ev.set_steer_at(t, best_s)
      ev.restore(snap)
      ev.step()
    ev.reset()
    best_total = ev.roll_cost()['total_cost']
    if best_total <= stop_at:
      return best_total
  ev.reset()
  return ev.roll_cost()['total_cost']


def optimize_segment_steer(model, data_path: str) -> dict:
  df = pd.read_csv(data_path)
  target = df['targetLateralAcceleration'].values[CONTROL_START_IDX:COST_END_IDX]
  ref_plan = solve_lataccel_qp(target)
  ev = SteerEvaluator(data_path, model)

  best_steer = pid_warmstart(model, data_path)
  ev.set_steer(best_steer)
  ev.reset()
  best_cost = ev.roll_cost()['total_cost']

  ev.set_steer(best_steer)
  shoot_steer_toward_plan(ev, ref_plan)
  shoot_cost = ev.roll_cost()['total_cost']
  if shoot_cost < best_cost:
    best_steer = ev.steer.copy()
    best_cost = shoot_cost

  ev.set_steer(best_steer)
  best_cost = coordinate_descent(ev)

  fp = fingerprint_from_csv(data_path)
  return {
    'steers': ev.steer.copy().astype(np.float32),
    'cost': float(best_cost),
    'fingerprint': fp,
  }
