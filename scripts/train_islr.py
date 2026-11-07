"""Isolated sign recognition on ASL Citizen or WLASL, starting from a pretrained encoder (or evaluating a trained classifier).

Each clip is cut to its recognition span (metadata field islr_span: the hand-visible frames +- 2, after the annotated span
for uncut WLASL files), standardised with the encoder's statistics and truncated to 1,032 frames. Training: 40 epochs,
batch 128, AdamW (lr 3e-4 for encoder and head, weight decay 0.05), 5 % warmup + cosine, label smoothing 0.1,
random temporal crop keeping >= 70 % of the frames, frame dropout 0.1. The epoch with the best validation top-1 is tested.

  python scripts/train_islr.py --data_root <release> --dataset asl_citizen \
      --encoder <release>/weights/mogloss/setting_b/pretraining/encoder_mmm_joint.pt --out_dir runs/islr_aslc
  python scripts/train_islr.py --data_root <release> --dataset wlasl100 --encoder ... --out_dir runs/islr_wlasl100
  python scripts/train_islr.py --data_root <release> --dataset asl_citizen \
      --eval_ckpt <release>/weights/mogloss/islr/asl_citizen/mmm_joint.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mogloss.data import ClipStore, load_metadata
from mogloss.features import FRAME_DIM
from mogloss.models import Classifier, load_encoder_checkpoint


def load_clips(root, dataset, max_items=0):
    """-> clips, glosses, splits, frame features (float16, as used in training)."""
    src = 'wlasl2000' if dataset.startswith('wlasl') else dataset
    meta = load_metadata(root, [src])
    store = ClipStore(root, 'b', [src])
    rows = [r for r in meta.values() if dataset != 'wlasl100' or r.get('in_wlasl100')]
    if max_items:  # quick test: the first max_items clips of each split
        rows = [r for s in ('train', 'val', 'test') for r in [x for x in rows if x['split'] == s][:max_items]]
    clips, gloss, split, feats = [], [], [], []
    for n, r in enumerate(rows):
        lo, hi = r['islr_span']
        clips.append(r['id'])
        gloss.append(r['gloss'])
        split.append(r['split'])
        feats.append(store.features(r['id'])[lo:hi].astype(np.float16))
        if n % 10000 == 0:
            print('loading', n, len(rows), flush=True)
    return clips, np.array(gloss), np.array(split), feats


def batchify(feats, idx, mu, sd, max_len, train=False, step_drop=0.0, rng=None):
    xs = []
    for i in idx:
        f = (feats[i].astype(np.float32) - mu) / sd
        if train and len(f) > 6:
            keep = int(np.ceil(len(f) * rng.uniform(0.7, 1.0)))
            s = rng.integers(0, len(f) - keep + 1)
            f = f[s:s + keep]
        xs.append(torch.from_numpy(f[:max_len]))
    L = max(len(t) for t in xs)
    x = torch.zeros(len(xs), L, xs[0].shape[1])
    m = torch.zeros(len(xs), L, dtype=torch.bool)
    for j, t in enumerate(xs):
        x[j, :len(t)], m[j, :len(t)] = t, True
    if train and step_drop > 0:
        x = x.masked_fill((torch.rand(x.shape[:2]) < step_drop).unsqueeze(-1), 0.0)
    return x, m


@torch.no_grad()
def evaluate(model, feats, idx, y, mu, sd, max_len, dev, bs=256):
    model.eval()
    c1 = c5 = 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x, m = batchify(feats, b, mu, sd, max_len)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(x.to(dev), m.to(dev)).float()
        t5 = lg.topk(5, dim=1).indices.cpu()
        yy = torch.from_numpy(y[b])
        c1 += int((t5[:, 0] == yy).sum())
        c5 += int((t5 == yy[:, None]).any(1).sum())
    model.train()
    return c1 / len(idx), c5 / len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--dataset', choices=['asl_citizen', 'wlasl100', 'wlasl2000'], required=True)
    ap.add_argument('--encoder', default='', help='pretrained encoder checkpoint (training)')
    ap.add_argument('--eval_ckpt', default='', help='trained classifier checkpoint (evaluation only)')
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lr_enc', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--label_smooth', type=float, default=0.1)
    ap.add_argument('--step_drop', type=float, default=0.1)
    ap.add_argument('--max_len', type=int, default=1032)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_items', type=int, default=0, help='quick test: clips per split')
    a = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    clips, gl, sp, feats = load_clips(a.data_root, a.dataset, a.max_items)

    if a.eval_ckpt:
        ck = torch.load(a.eval_ckpt, map_location='cpu', weights_only=False)
        classes, c = ck['classes'], ck['config']
        mu, sd = np.asarray(ck['mu'], np.float32), np.asarray(ck['sd'], np.float32)
        model = Classifier(len(classes), c.get('din', FRAME_DIM), c['d'], c['layers'], c['heads'], c['max_len'])
        model.load_state_dict(ck['model'])
    else:
        classes = sorted(set(gl[sp == 'train'].tolist()))
        enc, mu, sd, c = load_encoder_checkpoint(a.encoder)
        model = Classifier(len(classes), c.get('din', FRAME_DIM), c['d'], c['layers'], c['heads'], c['max_len'])
        model.enc.load_state_dict(enc)
    model = model.to(dev)
    ci = {g: k for k, g in enumerate(classes)}
    y = np.array([ci.get(g, -1) for g in gl])
    ok = (y >= 0) & (np.array([len(f) for f in feats]) >= 2)
    idx = {s: np.where((sp == s) & ok)[0] for s in ('train', 'val', 'test')}
    print({s: len(v) for s, v in idx.items()}, 'classes', len(classes), flush=True)

    if a.eval_ckpt:
        t1, t5 = evaluate(model, feats, idx['test'], y, mu, sd, a.max_len, dev)
        print(json.dumps({'test_top1': t1, 'test_top5': t5}), flush=True)
        return

    os.makedirs(a.out_dir, exist_ok=True)
    opt = torch.optim.AdamW([{'params': model.enc.parameters(), 'lr': a.lr_enc}, {'params': model.head.parameters(), 'lr': a.lr}], weight_decay=a.wd)
    base = [g['lr'] for g in opt.param_groups]
    steps = a.epochs * (len(idx['train']) // a.batch_size)
    warm = max(1, int(0.05 * steps))
    cfg = {k: c[k] for k in ('d', 'layers', 'heads', 'max_len')} | {'din': FRAME_DIM}
    best, step = -1.0, 0
    for ep in range(a.epochs):
        perm = rng.permutation(idx['train'])
        tl = tn = 0
        for i in range(0, len(perm) - a.batch_size + 1, a.batch_size):
            f = step / warm if step < warm else 0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, steps - warm)))
            for g, b0 in zip(opt.param_groups, base):
                g['lr'] = b0 * f
            b = perm[i:i + a.batch_size]
            x, m = batchify(feats, b, mu, sd, a.max_len, True, a.step_drop, rng)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(x.to(dev), m.to(dev))
            loss = F.cross_entropy(lg.float(), torch.from_numpy(y[b]).to(dev), label_smoothing=a.label_smooth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tl, tn = tl + float(loss.detach()), tn + 1
        v1, v5 = evaluate(model, feats, idx['val'], y, mu, sd, a.max_len, dev)
        print(f'ep {ep + 1}/{a.epochs} loss {tl / max(tn, 1):.4f} VAL top1 {v1:.4f} top5 {v5:.4f}', flush=True)
        if v1 > best:
            best = v1
            torch.save({'model': model.state_dict(), 'classes': classes, 'config': cfg, 'mu': mu, 'sd': sd, 'epoch': ep + 1},
                       os.path.join(a.out_dir, 'best.pt'))
    ck = torch.load(os.path.join(a.out_dir, 'best.pt'), map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    t1, t5 = evaluate(model, feats, idx['test'], y, mu, sd, a.max_len, dev)
    res = {'best_epoch': ck['epoch'], 'test_top1': t1, 'test_top5': t5}
    json.dump(res, open(os.path.join(a.out_dir, 'result.json'), 'w'), indent=1)
    print('TEST', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
