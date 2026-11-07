"""Translation training: pretraining from an encoder checkpoint, or per-dataset fine-tuning from a translation checkpoint.

Model: motion encoder -> Linear -> T5-base encoder -> T5-base decoder. Captions are tokenised with the T5 tokenizer
(at most 64 tokens). AdamW (weight decay 0.01), cosine schedule with warmup, gradient clipping 1.0, batch 48.

Pretraining (keeps the checkpoint with the best BLEU-4 on the 600 validation clips):
  python scripts/train_translation.py --mode pretrain --data_root <release> --setting b \
      --encoder runs/joint_b/encoder.pt --train_list splits/setting_b/translation_train.txt \
      --val_list splits/setting_b/val_600.txt --steps 196100 --out_dir runs/tr_b
  Setting A: --setting a --train_list splits/setting_a/translation_train.txt
      --val_list splits/setting_a/pretrain_youtube_asl_dev_600.txt --steps 250935 (20 epochs)
Fine-tuning (setting A, 20 epochs, reports the last checkpoint):
  python scripts/train_translation.py --mode finetune --data_root <release> --setting a --init runs/tr_a/best.pt \
      --train_list splits/setting_a/finetune_how2sign_train.txt --test_list splits/setting_a/test_how2sign.txt \
      --steps 12935 --out_dir runs/ft_a_how2sign
  (STEM 24630 steps, OpenASL 38457 steps; 20 epochs = ceil(20 x training clips / 48))
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mogloss.data import ClipStore, caption, load_metadata, read_list
from mogloss.models import Translator, load_encoder_checkpoint
from mogloss.translation import ClipSet, collate, micro_step, score, translate

DEFAULTS = {'pretrain': dict(lr=1e-4, lr_enc=5e-5, warmup=2000, val_every=10000),
            'finetune': dict(lr=5e-5, lr_enc=2.5e-5, warmup=500, val_every=2500)}


def build_model(a):
    if a.mode == 'pretrain':
        enc, mu, sd, c = load_encoder_checkpoint(a.encoder)
        model = Translator(c.get('din', 207), c['d'], c['layers'], c['heads'], c['max_len'], mu, sd)
        model.enc.load_state_dict(enc)
        cfg = {k: c[k] for k in ('din', 'd', 'layers', 'heads', 'max_len')}
    else:
        ck = torch.load(a.init, map_location='cpu', weights_only=False)
        c = ck['config']
        model = Translator(c.get('din', 207), c['d'], c['layers'], c['heads'], c['max_len'])
        model.load_state_dict(ck['model'], strict=True)
        cfg = {k: c[k] for k in ('din', 'd', 'layers', 'heads', 'max_len')}
    return model, cfg | {'text_model': 't5-base'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pretrain', 'finetune'], required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--setting', choices=['a', 'b'], required=True)
    ap.add_argument('--encoder', default='', help='encoder checkpoint (pretrain)')
    ap.add_argument('--init', default='', help='translation checkpoint (finetune)')
    ap.add_argument('--train_list', required=True)
    ap.add_argument('--val_list', default='', help='default (finetune): the training list')
    ap.add_argument('--test_list', default='')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--batch_size', type=int, default=48)
    ap.add_argument('--lr', type=float)
    ap.add_argument('--lr_enc', type=float)
    ap.add_argument('--warmup', type=int)
    ap.add_argument('--val_every', type=int)
    ap.add_argument('--n_val', type=int, default=600)
    ap.add_argument('--max_tokens', type=int, default=12288, help='split batches with more padded frames (same gradient)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    for k, v in DEFAULTS[a.mode].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    log = open(os.path.join(a.out_dir, 'train.log'), 'a')

    def P(*s):
        msg = ' '.join(str(x) for x in s)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    rel = lambda p: p if os.path.isabs(p) else os.path.join(a.data_root, p)
    store = ClipStore(a.data_root, a.setting)
    meta = load_metadata(a.data_root)
    mk = lambda path: [(c, caption(meta, c, a.setting)) for c in read_list(rel(path)) if c in store and c in meta and caption(meta, c, a.setting)]
    tr = mk(a.train_list)
    va = mk(a.val_list or a.train_list)
    random.Random(0).shuffle(va)
    va = va[:a.n_val]
    te = mk(a.test_list) if a.test_list else []
    P(f'{a.mode}: train {len(tr)} val {len(va)} test {len(te)}')

    tok = AutoTokenizer.from_pretrained('t5-base')
    model, cfg = build_model(a)
    model = model.to(dev)
    groups = [{'params': list(model.enc.parameters()), 'lr': a.lr_enc},
              {'params': list(model.proj.parameters()) + list(model.t5.parameters()), 'lr': a.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base = [g['lr'] for g in groups]
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    dl = DataLoader(ClipSet(store, tr), a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True,
                    persistent_workers=a.workers > 0)
    save = lambda name, step, val=None: torch.save({'model': model.state_dict(), 'config': cfg, 'step': step, 'val': val}, os.path.join(a.out_dir, name))

    best = -1.0
    if a.mode == 'finetune':
        p, r, _ = translate(model, store, va, tok, dev)
        best = score(p, r)['bleu4']
        save('best.pt', 0)
    step, t0, agg = 0, time.time(), []
    while step < a.steps:
        for x, m, t, _ in dl:
            for g, b0 in zip(opt.param_groups, base):
                g['lr'] = b0 * sched(step)
            lab = tok(t, padding=True, truncation=True, max_length=64, return_tensors='pt').input_ids
            lab[lab == tok.pad_token_id] = -100
            opt.zero_grad(set_to_none=True)
            agg.append(micro_step(model, x, m, lab, dev, a.max_tokens))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                P(f'step {step} loss {np.mean(agg):.4f} | {(time.time() - t0) / step:.2f}s/it')
                agg = []
            if step % a.val_every == 0 or step == a.steps:
                p, r, _ = translate(model, store, va, tok, dev)
                s = score(p, r)
                P(f'VAL step {step} BLEU-4 {s["bleu4"]:.2f} ROUGE-L {s["rougeL"]:.2f}')
                if s['bleu4'] > best:
                    best = s['bleu4']
                    save('best.pt', step, s)
            if step >= a.steps:
                break
    save('last.pt', step)

    if te:
        # pretraining reports the best validation checkpoint, fine-tuning the last one
        if a.mode == 'pretrain':
            model.load_state_dict(torch.load(os.path.join(a.out_dir, 'best.pt'), map_location='cpu', weights_only=False)['model'])
        p, r, c = translate(model, store, te, tok, dev)
        s = score(p, r)
        P('TEST ' + json.dumps(s))
        json.dump(s, open(os.path.join(a.out_dir, 'test_scores.json'), 'w'), indent=1)
        with open(os.path.join(a.out_dir, 'test_predictions.tsv'), 'w') as f:
            f.write('clip\tref\thyp\n')
            for cc, rr, pp in zip(c, r, p):
                f.write(f'{cc}\t{rr}\t{pp}\n')


if __name__ == '__main__':
    main()
