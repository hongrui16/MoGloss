"""Export the motion encoder of a stage-2 joint checkpoint in the stage-1 format (used by translation and recognition).

The joint model adds a motion modality embedding to every frame token; it is folded into the input bias, which is exact
for unmasked frames. Pseudo-gloss parts are dropped.

  python scripts/export_encoder.py --joint runs/joint_b/best.pt --out runs/joint_b/encoder.pt
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--joint', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    jk = torch.load(a.joint, map_location='cpu', weights_only=False)
    sd = {k[2:]: v.clone() for k, v in jk['model'].items() if k.startswith('m.')}
    sd['enc.inp.bias'] = sd['enc.inp.bias'] + jk['model']['mod.weight'][1].to(sd['enc.inp.bias'].dtype)
    cfg = {k: v for k, v in jk['config'].items() if k != 'pseudo_gloss_vocab'}
    mu, sdv = jk['model']['mu'].numpy(), jk['model']['sd'].numpy()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({'model': sd, 'config': cfg, 'mu': mu, 'sd': sdv, 'step': jk.get('step')}, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
