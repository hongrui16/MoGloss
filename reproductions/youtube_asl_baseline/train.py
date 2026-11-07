"""YouTube-ASL baseline (Uthus et al., NeurIPS 2023 Datasets and Benchmarks): keypoints -> linear layer -> T5-v1.1-Base.

Recipe of the paper, with our RTMPose 2D keypoints as input:
  input   85 points per frame: both hands 21 + 21, body 6 (shoulders, elbows, hips), face 37 (eyebrows, eyes, outer lips,
          outline). COCO-WholeBody indices: body 5, 6, 7, 8, 11, 12; hands 91-132; face (iBUG 68 = WholeBody 23-90) brows 17-26,
          eyes 36-47, outer lips 48-59, outline 0 / 8 / 16. A point with score < 0.3 is missing and set to -10. The visible points
          of a clip are min-max scaled per axis over the whole clip. At most 256 frames (longer clips are truncated).
          The paper uses MediaPipe x / y / z (255 values per frame); here x / y (170 values).
  model   a learned linear projection of each frame into T5-v1.1-Base (pretrained), decoder context 128 tokens.
  train   Adafactor, constant lr 1e-3, batch 128. Pretraining on YouTube-ASL with validation every --val_every steps and
          early stopping (--patience, --min_gain), best.pt = best dev BLEU. Fine-tuning for a fixed number of steps; the last
          step decodes the test split.
  decode  beam search (--beams), at most 128 new tokens; corpus BLEU from sacreBLEU (13a, case-sensitive).
Resumable: <run_dir>/last.pt (model + optimiser + step) every --save_every steps.
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
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.optimization import Adafactor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from release_data import read_split, load_captions, rtmpose_index  # noqa: E402

FACE = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 0, 8, 16]
POINTS = np.array([5, 6, 7, 8, 11, 12] + list(range(91, 133)) + [23 + i for i in FACE])
assert len(POINTS) == 85
MISSING = -10.0


def featurize(K, S, max_frames=256, thr=0.3):
    K, S = K[:max_frames, POINTS], S[:max_frames, POINTS]
    ok = S >= thr
    x = np.full(K.shape, MISSING, np.float32)
    if ok.any():
        p = K[ok]
        lo, hi = p.min(0), p.max(0)
        x[ok] = (K[ok] - lo) / np.maximum(hi - lo, 1e-6)   # unit bounding box over the clip
    return x.reshape(len(x), -1)                           # (T, 170)


class ClipSet(Dataset):
    def __init__(self, items, index):
        self.items = items
        self.index = index

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, text = self.items[i]
        z = self.index.load(cid)
        K, S = np.asarray(z['keypoints'], np.float32), np.asarray(z['scores'], np.float32)
        return torch.from_numpy(featurize(K, S)), text, cid


def collate(b):
    L = max(len(x) for x, _, _ in b)
    x = torch.zeros(len(b), L, b[0][0].shape[1])
    m = torch.zeros(len(b), L, dtype=torch.long)
    for i, (xx, _, _) in enumerate(b):
        x[i, :len(xx)] = xx
        m[i, :len(xx)] = 1
    return x, m, [t for _, t, _ in b], [c for _, _, c in b]


class YTASLBaseline(nn.Module):
    def __init__(self, t5):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5)
        self.proj = nn.Linear(len(POINTS) * 2, self.t5.config.d_model)

    def forward(self, x, m, labels):
        return self.t5(inputs_embeds=self.proj(x), attention_mask=m, labels=labels).loss

    @torch.no_grad()
    def generate(self, x, m, tok, beams=5):
        out = self.t5.generate(inputs_embeds=self.proj(x), attention_mask=m, num_beams=beams, max_new_tokens=128)
        return tok.batch_decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='downloaded MoGloss release')
    ap.add_argument('--cache_dir', default='', help='where the tar shard indexes are cached (default ~/.cache/mogloss_index)')
    ap.add_argument('--train_split', required=True, help='e.g. setting_a/pretrain_youtube_asl_train')
    ap.add_argument('--val_split', default='')
    ap.add_argument('--test_split', default='')
    ap.add_argument('--t5', default='google/t5-v1_1-base')
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--init', default='')
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--val_every', type=int, default=10000)
    ap.add_argument('--save_every', type=int, default=2000)
    ap.add_argument('--n_val', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--beams', type=int, default=5)
    ap.add_argument('--patience', type=int, default=0, help='stop when the best dev BLEU gained < --min_gain over this many steps')
    ap.add_argument('--min_gain', type=float, default=0.0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = 'cuda'
    os.makedirs(a.run_dir, exist_ok=True)
    log = open(os.path.join(a.run_dir, 'train.log'), 'a')

    def P(*s):
        msg = ' '.join(str(x) for x in s)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    index = rtmpose_index(a.data_root, 'setting_a', a.cache_dir or None)
    caps = {c: ' '.join(t.split()) for c, t in load_captions(a.data_root, 'setting_a').items()}

    def items_of(split):
        return [(c, caps[c]) for c in read_split(a.data_root, split) if caps.get(c) and c in index]

    tok = AutoTokenizer.from_pretrained(a.t5)
    tr = items_of(a.train_split)
    va = items_of(a.val_split)[:a.n_val] if a.val_split else []
    te = items_of(a.test_split) if a.test_split else []
    P(f'train {len(tr)} val {len(va)} test {len(te)} | {a.t5} + Linear({len(POINTS) * 2}) | batch {a.batch_size} lr {a.lr} Adafactor | steps {a.steps}')
    model = YTASLBaseline(a.t5).to(dev)
    opt = Adafactor(model.parameters(), lr=a.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    step = 0
    last = os.path.join(a.run_dir, 'last.pt')
    if os.path.exists(last):
        ck = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        step = ck['step']
        P(f'resumed {last} at step {step}')
    elif a.init:
        model.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=False)['model'])
        P(f'init from {a.init}')
    dl = DataLoader(ClipSet(tr, index), a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True,
                    persistent_workers=a.workers > 0)

    def run_eval(items):
        import sacrebleu
        model.eval()
        preds, refs, ids = [], [], []
        for x, m, t, c in DataLoader(ClipSet(items, index), 64, shuffle=False, num_workers=4, collate_fn=collate):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                preds += model.generate(x.to(dev), m.to(dev), tok, a.beams)
            refs += t
            ids += c
        model.train()
        return sacrebleu.corpus_bleu(preds, [refs]), preds, refs, ids

    def save():
        tmp = last + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step}, tmp)
        os.replace(tmp, last)

    t0 = time.time()
    agg, hist, best_val, stop = [], [], -1.0, False
    hp = os.path.join(a.run_dir, 'val_history.json')
    if os.path.exists(hp):
        hist = [tuple(x) for x in json.load(open(hp))]
        best_val = max([v for _, v in hist] or [-1.0])
    while step < a.steps and not stop:
        for x, m, t, _ in dl:
            lab = tok(t, padding=True, truncation=True, max_length=128, return_tensors='pt').input_ids
            lab[lab == tok.pad_token_id] = -100
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(x.to(dev), m.to(dev), lab.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            agg.append(loss.item())
            if step % 100 == 0:
                P(f'step {step} loss {np.mean(agg):.4f} | {(time.time() - t0) / max(1, len(agg)):.2f}s/it')
                agg = []
                t0 = time.time()
            if step % a.save_every == 0 or step == a.steps:
                save()
            if va and (step % a.val_every == 0 or step == a.steps):
                b, p, r, _ = run_eval(va)
                P(f'VAL step {step} BLEU {b.score:.2f} | {b}')
                for i in range(2):
                    P(f'  ref: {r[i]}\n  hyp: {p[i]}')
                hist.append((step, b.score))
                if b.score > best_val:
                    best_val = b.score
                    torch.save({'model': model.state_dict(), 'step': step, 'val': b.score}, os.path.join(a.run_dir, 'best.pt'))
                    P(f'  -> new best {best_val:.2f}')
                if a.patience:
                    old = max([v for s_, v in hist if s_ <= step - a.patience] or [-1.0])
                    if old >= 0 and max(v for _, v in hist) - old < a.min_gain:
                        P(f'EARLY STOP at step {step}: best dev {max(v for _, v in hist):.2f} gained < {a.min_gain} over the last {a.patience} steps (was {old:.2f})')
                        save()
                        stop = True
            if va and hist:
                json.dump(hist, open(hp, 'w'))
            if step >= a.steps or stop:
                break
    if te:
        from sacrebleu.metrics import BLEU
        b, p, r, ids = run_eval(te)
        b14 = [round(BLEU(max_ngram_order=n).corpus_score(p, [r]).score, 2) for n in (1, 2, 3, 4)]
        P(f'TEST LAST (step {step}) BLEU-1/2/3/4 {b14} | {b}')
        json.dump({'n': len(r), 'bleu_1to4': b14, 'step': step}, open(os.path.join(a.run_dir, 'test_last.json'), 'w'), indent=1)
        with open(os.path.join(a.run_dir, 'test_predictions.tsv'), 'w') as f:
            f.write('clip_id\tref\thyp\n')
            for c_, r_, p_ in zip(ids, r, p):
                f.write(f'{c_}\t{r_}\t{p_}\n')
    P('DONE', a.run_dir)


if __name__ == '__main__':
    main()
