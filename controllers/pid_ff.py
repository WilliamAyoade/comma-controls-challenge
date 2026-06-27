from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  PID with feedforward, preview, and roll compensation.
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.k_ff = 0.22
    self.k_preview = 0.05
    self.k_roll = 0.10
    self.error_integral = 0
    self.prev_error = 0
    self.prev_target = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = (target_lataccel - current_lataccel)
    self.error_integral = np.clip(self.error_integral + error, -10.0, 10.0)
    error_diff = error - self.prev_error
    self.prev_error = error

    target_rate = target_lataccel - self.prev_target
    self.prev_target = target_lataccel

    preview = 0.0
    if future_plan.lataccel:
      preview = future_plan.lataccel[min(9, len(future_plan.lataccel) - 1)] - target_lataccel

    pid = self.p * error + self.i * self.error_integral + self.d * error_diff
    ff = self.k_ff * target_lataccel + self.k_preview * preview
    roll = -self.k_roll * state.roll_lataccel
    return pid + ff + roll
