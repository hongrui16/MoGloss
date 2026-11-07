"""Randomly initialised encoder in the stage-1 checkpoint format (the "no pretraining" baseline, and the starting point
of the "joint only" ablation). Frame statistics are copied from a stage-1 MMM checkpoint so the inputs are scaled the same way.

  python scripts/init_random_encoder.py --stats_from runs/mmm_b/best.pt --out runs/random_b/encoder.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mogloss.models import MMM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats_from', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    ck = torch.load(a.stats_from, map_location='cpu', weights_only=False)
    c = ck['config']
    torch.manual_seed(a.seed)
    model = MMM(c['din'], c['d'], c['layers'], c['heads'], c['max_len'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': c, 'mu': ck['mu'], 'sd': ck['sd']}, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
