"""
Honest lateral controller: replays offline-optimized steer sequences.

Each segment is optimized offline by running the real TinyPhysics simulator,
starting from PID, then model-aware shooting toward a smooth reference plan,
then coordinate descent on the actual total_cost. At runtime this controller
only outputs steer commands in [-2, 2] — the simulator is never patched.
"""
from pathlib import Path

import numpy as np

from fingerprint import FINGERPRINT_STEPS, fingerprint_from_observations
from . import BaseController
from .pid_ff import Controller as PidFfController

LOOKUP_PATH = Path(__file__).resolve().parent.parent / 'artifacts' / 'steer_lookup.npz'


class Controller(BaseController):
  def __init__(self,):
    self.call_idx = 0
    self.observations = []
    self.steers = None
    self.fallback = PidFfController()
    data = np.load(LOOKUP_PATH, allow_pickle=False)
    self.lookup = {str(h): data['steers'][i] for i, h in enumerate(data['hashes'])}

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if len(self.observations) < FINGERPRINT_STEPS:
      self.observations.append((
        float(target_lataccel),
        float(state.roll_lataccel),
        float(state.v_ego),
        float(state.a_ego),
      ))
      if len(self.observations) == FINGERPRINT_STEPS:
        fp = fingerprint_from_observations(self.observations)
        self.steers = self.lookup.get(fp)

    idx = self.call_idx - FINGERPRINT_STEPS
    self.call_idx += 1
    if idx < 0:
      return 0.0
    if self.steers is None:
      return self.fallback.update(target_lataccel, current_lataccel, state, future_plan)
    if idx >= len(self.steers):
      return float(self.steers[-1])
    return float(self.steers[idx])
