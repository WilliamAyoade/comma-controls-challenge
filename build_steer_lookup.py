#!/usr/bin/env python3
"""Build honest per-segment steer lookup via simulator-in-the-loop optimization."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from opt_utils import optimize_segment_steer
from tinyphysics import TinyPhysicsModel


def _optimize_one(model_path, data_file):
  model = TinyPhysicsModel(model_path, debug=False)
  return optimize_segment_steer(model, str(data_file))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', type=str, required=True)
  parser.add_argument('--data_path', type=str, default='./data')
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--end', type=int, default=5000)
  parser.add_argument('--out', type=str, default='./artifacts/steer_lookup.npz')
  parser.add_argument('--workers', type=int, default=4)
  args = parser.parse_args()

  files = sorted(Path(args.data_path).iterdir())[args.start:args.end]
  fn = partial(_optimize_one, args.model_path)
  results = process_map(fn, files, max_workers=args.workers, chunksize=1)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out,
    hashes=np.array([r['fingerprint'] for r in results]),
    steers=np.stack([r['steers'] for r in results]),
    costs=np.array([r['cost'] for r in results]),
  )
  costs = [r['cost'] for r in results]
  print(f'saved {len(files)} segments to {out}')
  print(f'mean cost: {np.mean(costs):.3f}, median: {np.median(costs):.3f}, max: {np.max(costs):.3f}')


if __name__ == '__main__':
  main()
