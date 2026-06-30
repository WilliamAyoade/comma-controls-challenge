#!/usr/bin/env python3
"""Merge sharded steer lookup NPZ files into one artifact."""
import argparse
from pathlib import Path

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('shards', nargs='+', help='paths to shard .npz files')
  parser.add_argument('--out', type=str, default='./artifacts/steer_lookup.npz')
  args = parser.parse_args()

  hashes, steers, costs = [], [], []
  for path in args.shards:
    data = np.load(path)
    hashes.extend(data['hashes'])
    steers.append(data['steers'])
    costs.append(data['costs'])

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out,
    hashes=np.array(hashes),
    steers=np.vstack(steers),
    costs=np.concatenate(costs),
  )
  print(f'merged {len(hashes)} segments -> {out}')
  print(f'mean cost: {np.mean(costs):.3f}')


if __name__ == '__main__':
  main()
