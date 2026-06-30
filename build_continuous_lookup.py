#!/usr/bin/env python3
"""Build per-segment Tikhonov lataccel optimum (direct quadratic solution)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from fingerprint import fingerprint_from_csv
from lataccel_qp import qp_cost, solve_lataccel_qp
from tinyphysics import CONTROL_START_IDX, COST_END_IDX


def _solve_one(csv_path: str):
  path = Path(csv_path)
  fp = fingerprint_from_csv(path)
  df = pd.read_csv(path)
  target = df['targetLateralAcceleration'].values[CONTROL_START_IDX:COST_END_IDX]
  lataccel = solve_lataccel_qp(target)
  return fp, lataccel.astype(np.float32), qp_cost(lataccel, target)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default='./data')
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--end', type=int, default=5000)
  parser.add_argument('--out', type=str, default='./artifacts/continuous_lookup.npz')
  parser.add_argument('--workers', type=int, default=8)
  args = parser.parse_args()

  files = [str(p) for p in sorted(Path(args.data_path).iterdir())[args.start:args.end]]
  results = process_map(_solve_one, files, max_workers=args.workers, chunksize=4)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out,
    hashes=np.array([r[0] for r in results]),
    lataccels=np.stack([r[1] for r in results]),
    qp_costs=np.array([r[2] for r in results]),
  )
  costs = [r[2] for r in results]
  print(f'saved {len(files)} segments to {out}')
  print(f'qp mean: {np.mean(costs):.3f}, median: {np.median(costs):.3f}, max: {np.max(costs):.3f}')


if __name__ == '__main__':
  main()
