"""Stage 1: masked motion modeling (MMM).

Random windows of 256 keypoint frames, one token per frame. Frames are standardised with statistics of the first 40 training
batches. 30 % of the valid frames are replaced by a learned [MASK] vector after the input layer, and the model regresses the
standardised frames at those positions (L1). The checkpoint with the lowest validation loss is kept.

  python scripts/pretrain_mmm.py --data_root <MoGloss release> --setting b \
      --train_list splits/setting_b/train.txt --val_list splits/setting_b/val.txt --out_dir runs/mmm_b
Setting A uses splits/setting_a/pretrain_youtube_asl_train.txt and pretrain_youtube_asl_dev.txt.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mogloss.data import ClipStore, read_list
from mogloss.features import FRAME_DIM
from mogloss.models import MMM


class WindowSet(Dataset):
    """A random (train) or centred (validation) window of `window` frames; shorter clips are padded with their last frame."""

    def __init__(self, store, clips, window, train):
        self.store, self.clips, self.window, self.train = store, clips, window, train

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        f = self.store.features(self.clips[i])
        T, W = len(f), self.window
        if T >= W:
            s = random.randint(0, T - W) if self.train else (T - W) // 2
            f, mask = f[s:s + W], np.ones(W, bool)
        else:
            f = np.concatenate([f, np.repeat(f[-1:], W - T, 0)])
            mask = np.concatenate([np.ones(T, bool), np.zeros(W - T, bool)])
        return torch.from_numpy(f), torch.from_numpy(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--setting', choices=['a', 'b'], default='b')
    ap.add_argument('--train_list', required=True)
    ap.add_argument('--val_list', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--window', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=5000)
    ap.add_argument('--mask_p', type=float, default=0.3)
    ap.add_argument('--d', type=int, default=1024)
    ap.add_argument('--layers', type=int, default=16)
    ap.add_argument('--heads', type=int, default=16)
    ap.add_argument('--max_len', type=int, default=1032)
    ap.add_argument('--val_every', type=int, default=2000)
    ap.add_argument('--stat_batches', type=int, default=40)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    rel = lambda p: p if os.path.isabs(p) else os.path.join(a.data_root, p)
    store = ClipStore(a.data_root, a.setting)
    tr_ids = [c for c in read_list(rel(a.train_list)) if c in store]
    va_ids = [c for c in read_list(rel(a.val_list)) if c in store]
    tr, va = WindowSet(store, tr_ids, a.window, True), WindowSet(store, va_ids, a.window, False)
    dl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=a.workers > 0)
    dv = DataLoader(va, a.batch_size, shuffle=False, num_workers=min(4, a.workers))

    # per-dimension statistics over the valid frames of the first training batches
    s1 = s2 = n = 0.0
    for k, (x, m) in enumerate(dl):
        f = x[m].double()
        s1, s2, n = s1 + f.sum(0), s2 + (f ** 2).sum(0), n + len(f)
        if k + 1 >= a.stat_batches:
            break
    mu = (s1 / n).float().to(dev)
    sd = ((s2 / n - (s1 / n) ** 2).clamp_min(0).sqrt() + 1e-3).float().to(dev)

    model = MMM(FRAME_DIM, a.d, a.layers, a.heads, a.max_len).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    print(f'train {len(tr)} val {len(va)} | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M', flush=True)
    cfg = {'d': a.d, 'layers': a.layers, 'heads': a.heads, 'max_len': a.max_len, 'din': FRAME_DIM, 'mask_p': a.mask_p}

    def save(name, val, step):
        torch.save({'model': model.state_dict(), 'config': cfg, 'mu': mu.cpu().numpy(), 'sd': sd.cpu().numpy(), 'step': step, 'val': val},
                   os.path.join(a.out_dir, name))

    def loss_of(x, m, gen):
        xs = (x - mu) / sd
        sel = (torch.rand(m.shape, device=dev, generator=gen) < a.mask_p) & m
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = model(xs, m, sel)
        return ((y.float() - xs).abs().mean(-1) * sel).sum() / sel.sum().clamp_min(1)

    @torch.no_grad()
    def validate():
        model.eval()
        g = torch.Generator(device=dev)
        g.manual_seed(1234)
        tot = cnt = 0.0
        for x, m in dv:
            x, m = x.to(dev), m.to(dev)
            tot += float(loss_of(x, m, g)) * len(x)
            cnt += len(x)
        model.train()
        return tot / max(cnt, 1)

    best, step, t0, it = float('inf'), 0, time.time(), iter(dl)
    while step < a.steps:
        try:
            x, m = next(it)
        except StopIteration:
            it = iter(dl)
            x, m = next(it)
        f = step / a.warmup if step < a.warmup else 0.5 * (1 + np.cos(np.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))
        for g in opt.param_groups:
            g['lr'] = a.lr * f
        loss = loss_of(x.to(dev, non_blocking=True), m.to(dev), None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f'step {step} lr {a.lr * f:.2e} loss {float(loss):.4f} | {(time.time() - t0) / 100:.2f}s/it', flush=True)
            t0 = time.time()
        if step % a.val_every == 0 or step == a.steps:
            v = validate()
            print(f'VAL step {step} masked-L1 {v:.4f}', flush=True)
            save('last.pt', v, step)
            if v < best:
                best = v
                save('best.pt', v, step)
    json.dump({'best_val_masked_l1': best}, open(os.path.join(a.out_dir, 'result.json'), 'w'))


if __name__ == '__main__':
    main()
