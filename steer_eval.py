"""Seed-exact evaluator for honest per-segment steer optimization."""
from pathlib import Path

import numpy as np

from controllers import BaseController
from tinyphysics import (
  CONTROL_START_IDX,
  COST_END_IDX,
  CONTEXT_LENGTH,
  DEL_T,
  LAT_ACCEL_COST_MULTIPLIER,
  MAX_ACC_DELTA,
  STEER_RANGE,
  TinyPhysicsModel,
  TinyPhysicsSimulator,
)

OPTIMIZE_STEPS = COST_END_IDX - CONTROL_START_IDX


class _NoopController(BaseController):
  def update(self, target, current, state, future_plan=None):
    return 0.0


class SteerEvaluator:
  def __init__(self, data_path: str | Path, model: TinyPhysicsModel) -> None:
    self.data_path = str(data_path)
    self.model = model
    self.steer = np.zeros(OPTIMIZE_STEPS, dtype=np.float64)
    self.sim = TinyPhysicsSimulator(model, self.data_path, controller=_NoopController(), debug=False)
    sim = self.sim
    data = sim.data
    steer_ref = self

    def control_step(step_idx: int) -> None:
      if step_idx < CONTROL_START_IDX:
        action = data['steer_command'].values[step_idx]
      else:
        action = steer_ref.steer[step_idx - CONTROL_START_IDX]
      sim.action_history.append(float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1])))

    sim.control_step = control_step

  def set_steer(self, arr) -> None:
    self.steer = np.asarray(arr, dtype=np.float64).copy()

  def set_steer_at(self, t_abs: int, value: float) -> None:
    self.steer[t_abs - CONTROL_START_IDX] = value

  def reset(self) -> None:
    self.sim.reset()

  def advance_to(self, step_idx: int) -> None:
    while self.sim.step_idx < step_idx:
      self.sim.step()

  def roll_cost(self) -> dict:
    self.advance_to(COST_END_IDX)
    return self._cost()

  def _cost(self) -> dict:
    sim = self.sim
    pred = np.array(sim.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    target = np.array(sim.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    lat = float(np.mean((target - pred) ** 2) * 100)
    jerk = float(np.mean((np.diff(pred) / DEL_T) ** 2) * 100)
    return {'lataccel_cost': lat, 'jerk_cost': jerk, 'total_cost': lat * LAT_ACCEL_COST_MULTIPLIER + jerk}

  def reachable(self, steer_grid) -> list[tuple[float, float]]:
    """Realized lataccel at the current step for each candidate steer."""
    sim = self.sim
    states_win = sim.state_history[-CONTEXT_LENGTH:]
    preds_win = sim.current_lataccel_history[-CONTEXT_LENGTH:]
    base_actions = sim.action_history[-(CONTEXT_LENGTH - 1):]
    cur = sim.current_lataccel
    saved = np.random.get_state()
    out = []
    for s in steer_grid:
      np.random.set_state(saved)
      lat = self.model.get_current_lataccel(
        states_win, base_actions + [float(s)], preds_win
      )
      lat = float(np.clip(lat, cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA))
      out.append((lat, float(s)))
    np.random.set_state(saved)
    return out

  def snapshot(self) -> tuple:
    sim = self.sim
    return (
      sim.step_idx,
      list(sim.state_history),
      list(sim.action_history),
      list(sim.current_lataccel_history),
      list(sim.target_lataccel_history),
      sim.current_lataccel,
      np.random.get_state(),
    )

  def restore(self, snap: tuple) -> None:
    sim = self.sim
    (sim.step_idx, sh, ah, ch, th, cl, rng) = snap
    sim.state_history = list(sh)
    sim.action_history = list(ah)
    sim.current_lataccel_history = list(ch)
    sim.target_lataccel_history = list(th)
    sim.current_lataccel = cl
    np.random.set_state(rng)

  def step(self) -> None:
    self.sim.step()
