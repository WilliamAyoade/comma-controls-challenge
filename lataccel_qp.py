"""Closed-form Tikhonov optimum for the scored-window quadratic cost."""
import numpy as np
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER

N = COST_END_IDX - CONTROL_START_IDX
W_TRACK = LAT_ACCEL_COST_MULTIPLIER * 100.0 / N
W_JERK = 100.0 / (N - 1) / (DEL_T ** 2)

_L = np.zeros((N, N))
for k in range(N - 1):
  _L[k, k] += 1
  _L[k + 1, k + 1] += 1
  _L[k, k + 1] -= 1
  _L[k + 1, k] -= 1
MINV = np.linalg.inv(W_TRACK * np.eye(N) + W_JERK * _L)


def solve_lataccel_qp(target: np.ndarray) -> np.ndarray:
  target = np.asarray(target, dtype=np.float64)
  return MINV @ (W_TRACK * target)


def qp_cost(pred: np.ndarray, target: np.ndarray) -> float:
  lat = np.mean((target - pred) ** 2) * 100
  jerk = np.mean((np.diff(pred) / DEL_T) ** 2) * 100
  return float(lat * LAT_ACCEL_COST_MULTIPLIER + jerk)
