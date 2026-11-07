"""Stage 2: joint motion and pseudo-gloss masked modeling, initialised from the stage-1 MMM checkpoint.

One Transformer reads [pseudo-gloss units] + [keypoint frames]. Each sample is either a text sample (15-50 % of the units
masked, cross-entropy) or a motion sample (30-50 % of the frames masked, L1; in 10 % of the motion samples the units are
hidden). The checkpoint with the best masked-unit accuracy on the validation clips is kept.

  python scripts/pretrain_joint.py --data_root <MoGloss release> --setting b --init runs/mmm_b/best.pt \
      --lexicon preprocessing/pseudo_gloss/sign_lexicon.tsv --out_dir runs/joint_b
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mogloss.data import ClipStore, load_metadata, pseudo_gloss, read_list
from mogloss.models import MASK, PAD, JointModel, joint_loss
from mogloss.units import build_vocab, encode_units, read_lexicon

MAX_FRAMES = 1024


class JointSet(Dataset):
    """Each item draws one clip at random (a fixed draw per index when seed > 0, used for validation)."""

    def __init__(self, store, items, voc, n_items, seed=0):
        self.store, self.items, self.voc, self.n, self.seed = store, items, voc, n_items, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r = random.Random(self.seed * 7919 + i + (0 if self.seed else random.randrange(1 << 30)))
        clip, units = self.items[r.randrange(len(self.items))]
        x = self.store.features(clip)[:MAX_FRAMES]
        return torch.from_numpy(x), torch.tensor(encode_units(units, self.voc), dtype=torch.long)


def collate(b):
    L, T = max(len(x) for x, _ in b), max(len(u) for _, u in b)
    x = torch.zeros(len(b), L, b[0][0].shape[1])
    m = torch.zeros(len(b), L, dtype=torch.bool)
    u = torch.zeros(len(b), T, dtype=torch.long)
    for i, (xx, uu) in enumerate(b):
        x[i, :len(xx)], m[i, :len(xx)], u[i, :len(uu)] = xx, True, uu
    return x, m, u


@torch.no_grad()
def validate(model, loader, dev, seed=0):
    """Masked-unit accuracy with the clip's own motion and with another clip's motion; frame L1 with and without units."""
    g = torch.Generator(dev)
    g.manual_seed(seed)
    acc_true = acc_shuf = n = 0
    lw = lwo = 0.0
    k = 0
    for x, m, u in loader:
        x, m, u = x.to(dev), m.to(dev), u.to(dev)
        B, tm = len(x), u != PAD
        tsel = (torch.rand(u.shape, device=dev, generator=g) < 0.3) & tm
        tsel[torch.arange(B), tm.float().argmax(1)] |= tm.any(1)
        ui = torch.where(tsel, torch.full_like(u, MASK), u)
        none = torch.zeros_like(m)
        _, yt = model(x, m, ui, none)
        _, ys = model(torch.roll(x, 1, 0), torch.roll(m, 1, 0), ui, none)
        acc_true += (yt[tsel].argmax(-1) == u[tsel]).sum().item()
        acc_shuf += (ys[tsel].argmax(-1) == u[tsel]).sum().item()
        n += tsel.sum().item()
        msel = (torch.rand(m.shape, device=dev, generator=g) < 0.4) & m
        zs = (x - model.mu) / model.sd
        ym, _ = model(x, m, u, msel)
        ym0, _ = model(x, m, u, msel, torch.ones(B, dtype=torch.bool, device=dev))
        lw += (((ym.float() - zs).abs().mean(-1) * msel).sum() / msel.sum()).item()
        lwo += (((ym0.float() - zs).abs().mean(-1) * msel).sum() / msel.sum()).item()
        k += 1
    k = max(k, 1)
    return {'unit_acc': acc_true / max(n, 1), 'unit_acc_shuffled_motion': acc_shuf / max(n, 1),
            'motion_l1': lw / k, 'motion_l1_no_text': lwo / k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--setting', choices=['a', 'b'], default='b')
    ap.add_argument('--train_list', default='', help='default: splits/setting_<s>/joint_train.txt')
    ap.add_argument('--val_list', default='', help='default: splits/setting_<s>/joint_val.txt')
    ap.add_argument('--init', required=True, help='stage-1 MMM checkpoint')
    ap.add_argument('--lexicon', required=True, help='sign lexicon (its entries are added to the unit vocabulary)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--vocab', type=int, default=30000)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--steps', type=int, default=80000)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--warmup', type=int, default=5000)
    ap.add_argument('--val_every', type=int, default=4000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    sp = os.path.join(a.data_root, 'splits', f'setting_{a.setting}')
    rel = lambda p: p if os.path.isabs(p) else os.path.join(a.data_root, p)
    store = ClipStore(a.data_root, a.setting)
    meta = load_metadata(a.data_root)

    def items(path):
        out = []
        for c in read_list(path):
            u = pseudo_gloss(meta, c, a.setting)
            if u:
                out.append((c, u))
        return out

    tr = items(rel(a.train_list) if a.train_list else os.path.join(sp, 'joint_train.txt'))
    va = items(rel(a.val_list) if a.val_list else os.path.join(sp, 'joint_val.txt'))
    voc = build_vocab([u for _, u in tr], read_lexicon(a.lexicon), a.vocab)
    print(f'vocab {len(voc)} | train {len(tr)} val {len(va)}', flush=True)

    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    c = ck['config']
    model = JointModel(len(voc), c['din'], c['d'], c['layers'], c['heads'], c['max_len'], ck['mu'], ck['sd'])
    model.m.load_state_dict(ck['model'])
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    dl = DataLoader(JointSet(store, tr, voc, a.steps * a.batch_size), a.batch_size, num_workers=a.workers, collate_fn=collate,
                    persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    dv = DataLoader(JointSet(store, va, voc, min(len(va), 1024), seed=1), a.batch_size, num_workers=min(4, a.workers), collate_fn=collate)
    cfg = dict(c) | {'pseudo_gloss_vocab': sorted(voc, key=voc.get)}
    g = torch.Generator(dev)
    g.manual_seed(a.seed)
    t0, agg, best, step = time.time(), [0.0, 0.0, 0.0, 0], -1, 0
    for x, m, u in dl:
        step += 1
        for gp in opt.param_groups:
            gp['lr'] = a.lr * sched(step)
        x, m, u = x.to(dev, non_blocking=True), m.to(dev), u.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, lt, lm, acc = joint_loss(model, x, m, u, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        agg = [agg[0] + lt, agg[1] + lm, agg[2] + acc, agg[3] + 1]
        if step % 100 == 0:
            print(f'step {step} text_ce {agg[0] / agg[3]:.3f} motion_l1 {agg[1] / agg[3]:.4f} unit_acc {agg[2] / agg[3]:.3f} | '
                  f'{(time.time() - t0) / step:.2f}s/it', flush=True)
            agg = [0.0, 0.0, 0.0, 0]
        if step % a.val_every == 0 or step == a.steps:
            model.eval()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                val = validate(model, dv, dev)
            model.train()
            print(f'VAL step {step} ' + json.dumps({k: round(v, 4) for k, v in val.items()}), flush=True)
            out = {'model': model.state_dict(), 'config': cfg, 'step': step, 'val': val}
            torch.save(out, os.path.join(a.out_dir, 'last.pt'))
            if val['unit_acc'] > best:
                best = val['unit_acc']
                torch.save(out, os.path.join(a.out_dir, 'best.pt'))
        if step >= a.steps:
            break


if __name__ == '__main__':
    main()
