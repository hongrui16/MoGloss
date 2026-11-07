"""SHuBERT translation (our reproduction; the SHuBERT translation code is not released).

Model: released SHuBERT encoder (checkpoint_836_400000.pt) -> learned softmax-weighted sum of its 12 layer outputs ->
Linear(768, 1536) -> ByT5-Base (pretrained) through inputs_embeds. SHuBERT is fine-tuned with a 10x smaller learning rate.
Loss: cross-entropy with label smoothing 0.2. Optimiser: AdamW, weight decay 0.1, linear warmup then cosine decay, gradient-norm
clip 1.0, bf16 autocast. By default weight decay is applied to every parameter (runs A and C1); --hf_weight_decay applies it only
where the HuggingFace Trainer does, i.e. not to biases and normalization layers (run C2).
Decoding: beam search (--beams), at most --max_label byte tokens; BLEU = sacreBLEU corpus BLEU (13a, case-sensitive).

Data: split lists and setting-A captions of the MoGloss release; one stream file <streams>/<clip>.npz per clip
(streams_rtmpose.py for run A, streams_mediapipe.py for runs C1 / C2). Clips without a stream file or with an empty caption are
skipped and counted. At most --max_frames frames per clip (longer clips are truncated).
Checkpoints: <run_dir>/last.pt every --save_every updates (resumable), best.pt = best dev BLEU (ties go to the later one).
After training, the test list is decoded with the last weights -> test_predictions.tsv (clip_id / ref / hyp) and test_last.json.

  translation pretraining: python train.py --train_list setting_a/pretrain_youtube_asl_train --val_list setting_a/pretrain_youtube_asl_dev \
      --steps 250000 --micro_bs 8 --accum 16 --lr 5e-4 --warmup 10000 ...
  fine-tuning:             python train.py --init <pretrain run_dir>/best.pt --train_list setting_a/finetune_how2sign_train \
      --test_list setting_a/test_how2sign --epochs 20 --micro_bs 8 --accum 16 --lr 1e-4 --warmup 485 ...
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from release_data import read_split, load_captions  # noqa: E402

SHUBERT = {}   # filled by import_shubert()


def import_shubert(repo):
    """SHuBERT model classes from a checkout of the official repository (fairseq inside it)."""
    sys.path.insert(0, os.path.join(repo, 'fairseq'))
    from examples.shubert.models.shubert import SHubertModel, SHubertConfig
    SHUBERT['model'], SHUBERT['config'] = SHubertModel, SHubertConfig


def items_of(data_root, split, captions, streams, have):
    """[(clip, stream file, caption)] in list order, + number of clips without a stream file."""
    out, miss = [], 0
    for c in read_split(data_root, split):
        text = ' '.join(captions.get(c, '').split())
        if not text:
            continue
        f = c + '.npz'
        if f in have:
            out.append((c, os.path.join(streams, f), text))
        else:
            miss += 1
    return out, miss


class ClipSet(Dataset):
    def __init__(self, items, max_frames):
        self.items = items
        self.max_frames = max_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, path, text = self.items[i]
        d = np.load(path)
        T = min(int(d['T']), self.max_frames)
        x = {k: torch.from_numpy(np.asarray(d[k][:T], np.float32)) for k in ('face', 'lhand', 'rhand', 'body')}
        return x, text, cid


def collate(b):
    L = max(len(x['face']) for x, _, _ in b)
    L += L % 2
    src = {k: torch.zeros(len(b), L, b[0][0][k].shape[1]) for k in ('face', 'lhand', 'rhand', 'body')}
    pad = torch.ones(len(b), L, dtype=torch.bool)
    for i, (x, _, _) in enumerate(b):
        n = len(x['face'])
        pad[i, :n] = False
        for k in src:
            src[k][i, :n] = x[k]
    return src, pad, [t for _, t, _ in b], [c for _, _, c in b]


class SHuBERTSLT(nn.Module):
    def __init__(self, shubert_ckpt, byt5):
        super().__init__()
        Model, Config = SHUBERT['model'], SHUBERT['config']
        self.shubert = Model(Config())
        sd = torch.load(shubert_ckpt, map_location='cpu', weights_only=False)
        sd = sd.get('model', sd)
        r = self.shubert.load_state_dict(sd, strict=False)
        self.load_report = (len(r.missing_keys), len(r.unexpected_keys))
        self.shubert.remove_pretraining_modules()
        L = Config().encoder_layers
        self.layer_w = nn.Parameter(torch.zeros(L))
        self.t5 = T5ForConditionalGeneration.from_pretrained(byt5)
        self.proj = nn.Linear(Config().encoder_embed_dim, self.t5.config.d_model)

    def encode(self, src, pad):
        B, T = pad.shape
        z = torch.zeros(B, T, 1, device=pad.device)
        source = [{'face': src['face'][i], 'left_hand': src['lhand'][i], 'right_hand': src['rhand'][i], 'body_posture': src['body'][i],
                   'label_face': z[i], 'label_left_hand': z[i], 'label_right_hand': z[i], 'label_body_posture': z[i]} for i in range(B)]
        res = self.shubert.extract_features(source, padding_mask=pad, kmeans_labels=None, mask=False)
        H = torch.stack([lr[-1].transpose(0, 1) for lr in res['layer_results']], 0)   # (L, B, T, D), as SHuBERT's shubert_inference.py
        h = (torch.softmax(self.layer_w, 0).view(-1, 1, 1, 1) * H).sum(0)
        return self.proj(h), (~pad).long()

    def forward(self, src, pad, labels, smoothing=0.2):
        e, m = self.encode(src, pad)
        logits = self.t5(inputs_embeds=e, attention_mask=m, labels=labels).logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, label_smoothing=smoothing)

    @torch.no_grad()
    def generate(self, src, pad, tok, beams=4, max_len=384):
        e, m = self.encode(src, pad)
        out = self.t5.generate(inputs_embeds=e, attention_mask=m, num_beams=beams, max_new_tokens=max_len)
        return tok.batch_decode(out, skip_special_tokens=True)


def to_dev(src, pad, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in src.items()}, pad.to(dev, non_blocking=True)


def param_groups(model, a, P):
    """AdamW groups: SHuBERT at lr x shubert_lr_ratio, the rest at lr. With --hf_weight_decay, biases and normalization
    parameters get no weight decay (the parameter split of the HuggingFace Trainer)."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not a.hf_weight_decay:
        sh = [p for n, p in named if n.startswith('shubert.')]
        rest = [p for n, p in named if not n.startswith('shubert.')]
        return [{'params': rest, 'lr': a.lr, 'weight_decay': a.wd}, {'params': sh, 'lr': a.lr * a.shubert_lr_ratio, 'weight_decay': a.wd}]
    from transformers.trainer_pt_utils import get_parameter_names
    decay = set(get_parameter_names(model, [nn.LayerNorm], [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]))
    groups = []
    for lr_, is_sh in ((a.lr, False), (a.lr * a.shubert_lr_ratio, True)):
        for wd_ in (a.wd, 0.0):
            ps = [p for n, p in named if n.startswith('shubert.') == is_sh and ((n in decay) == (wd_ > 0))]
            if ps:
                groups.append({'params': ps, 'lr': lr_, 'weight_decay': wd_})
    nd = [n for n, p in named if n not in decay]
    P(f'weight decay {a.wd} on {sum(p.numel() for n, p in named if n in decay) / 1e6:.1f}M params; no decay on {len(nd)} tensors '
      f'({sum(p.numel() for n, p in named if n not in decay) / 1e6:.2f}M), e.g. {nd[:4]}')
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='downloaded MoGloss release (splits + metadata)')
    ap.add_argument('--streams', required=True, help='folder of <clip>.npz stream files')
    ap.add_argument('--shubert_repo', required=True, help='checkout of https://github.com/ShesterG/SHuBERT (fairseq built)')
    ap.add_argument('--shubert_ckpt', required=True, help='released SHuBERT weights, checkpoint_836_400000.pt')
    ap.add_argument('--byt5', default='google/byt5-base')
    ap.add_argument('--train_list', required=True)
    ap.add_argument('--val_list', default='')
    ap.add_argument('--test_list', default='')
    ap.add_argument('--steps', type=int, default=0, help='length of the schedule in updates; 0 = use --epochs')
    ap.add_argument('--epochs', type=int, default=0)
    ap.add_argument('--stop_at', type=int, default=0, help='stop after this many updates, keeping the schedule of --steps')
    ap.add_argument('--micro_bs', type=int, default=8)
    ap.add_argument('--accum', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--shubert_lr_ratio', type=float, default=0.1)
    ap.add_argument('--warmup', type=int, default=10000)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--hf_weight_decay', action='store_true', help='no weight decay on biases and normalization layers (run C2)')
    ap.add_argument('--smoothing', type=float, default=0.2)
    ap.add_argument('--max_frames', type=int, default=1024)
    ap.add_argument('--max_label', type=int, default=384)
    ap.add_argument('--beams', type=int, default=4)
    ap.add_argument('--init', default='')
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--val_every', type=int, default=10000)
    ap.add_argument('--save_every', type=int, default=2000)
    ap.add_argument('--n_val', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--max_items', type=int, default=0, help='debug: use only this many training clips')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    import_shubert(a.shubert_repo)
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
    tok = AutoTokenizer.from_pretrained(a.byt5)
    captions = load_captions(a.data_root, 'setting_a')
    have = {e.name for e in os.scandir(a.streams)}
    tr, m_tr = items_of(a.data_root, a.train_list, captions, a.streams, have)
    va, m_va = items_of(a.data_root, a.val_list, captions, a.streams, have) if a.val_list else ([], 0)
    te, m_te = items_of(a.data_root, a.test_list, captions, a.streams, have) if a.test_list else ([], 0)
    va = va[:a.n_val]
    if a.max_items:
        tr, va, te = tr[:a.max_items], va[:16], te[:16]
    per_epoch = len(tr) // (a.micro_bs * a.accum)
    total = a.steps or per_epoch * a.epochs
    P(f'train {len(tr)} (no stream file {m_tr}) val {len(va)} ({m_va}) test {len(te)} ({m_te}) | SHuBERT + layer-weighted sum + Linear + {a.byt5} | '
      f'update = {a.micro_bs} x {a.accum} utterances, {per_epoch} updates/epoch, total {total} | lr {a.lr} (SHuBERT x{a.shubert_lr_ratio}) '
      f'warmup {a.warmup} cosine wd {a.wd} ls {a.smoothing}')
    model = SHuBERTSLT(a.shubert_ckpt, a.byt5).to(dev)
    P(f'SHuBERT ckpt load: missing {model.load_report[0]} unexpected {model.load_report[1]}')
    opt = torch.optim.AdamW(param_groups(model, a, P))
    base = [g['lr'] for g in opt.param_groups]

    def set_lr(s):
        f = s / max(1, a.warmup) if s < a.warmup else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, total - a.warmup))))
        for g, b in zip(opt.param_groups, base):
            g['lr'] = b * f
    step = 0
    last = os.path.join(a.run_dir, 'last.pt')
    if os.path.exists(last):
        ck = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        step = ck['step']
        P(f'RESUMED {last} at step {step}')
    elif a.init:
        model.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=False)['model'])
        P(f'init from {a.init}')
    dl = DataLoader(ClipSet(tr, a.max_frames), a.micro_bs, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True,
                    persistent_workers=a.workers > 0)

    def run_eval(items):
        import sacrebleu
        model.eval()
        preds, refs, ids = [], [], []
        for src, pad, t, c in DataLoader(ClipSet(items, a.max_frames), 16, shuffle=False, num_workers=4, collate_fn=collate):
            src, pad = to_dev(src, pad, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                preds += model.generate(src, pad, tok, a.beams, a.max_label)
            refs += t
            ids += c
        model.train()
        return sacrebleu.corpus_bleu(preds, [refs]), preds, refs, ids

    def save(path=last):
        tmp = path + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict() if path == last else None, 'step': step}, tmp)
        os.replace(tmp, path)
    hp = os.path.join(a.run_dir, 'val_history.json')
    hist = [tuple(x) for x in json.load(open(hp))] if os.path.exists(hp) else []
    model.train()
    t0 = time.time()
    agg = []
    micro = 0
    stop = a.stop_at or total
    while step < stop:
        for src, pad, t, _ in dl:
            lab = tok(t, padding=True, truncation=True, max_length=a.max_label, return_tensors='pt').input_ids
            lab[lab == tok.pad_token_id] = -100
            src, pad = to_dev(src, pad, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(src, pad, lab.to(dev), a.smoothing)
            (loss / a.accum).backward()
            agg.append(loss.item())
            micro += 1
            if micro % a.accum:
                continue
            set_lr(step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                P(f'step {step} loss {np.mean(agg):.4f} lr {opt.param_groups[0]["lr"]:.2e} | {(time.time() - t0) / 100:.2f}s/it')
                agg = []
                t0 = time.time()
            if step % a.save_every == 0 or step == stop:
                save()
            if va and (step % a.val_every == 0 or step == stop):
                b, p, r, _ = run_eval(va)
                P(f'VAL step {step} BLEU {b.score:.2f} | distinct hypotheses {len(set(p))}/{len(p)} | {b}')
                for i in range(2):
                    P(f'  ref: {r[i]}\n  hyp: {p[i]}')
                hist.append((step, b.score))
                json.dump(hist, open(hp, 'w'))
                if b.score >= max(v for _, v in hist):
                    save(os.path.join(a.run_dir, 'best.pt'))
                    P(f'  -> new best {b.score:.2f}')
                t0 = time.time()
            if step >= stop:
                break
    if te:
        b, p, r, ids = run_eval(te)
        P(f'TEST LAST (step {step}) BLEU {b.score:.2f} | {b}')
        with open(os.path.join(a.run_dir, 'test_predictions.tsv'), 'w') as f:
            f.write('clip_id\tref\thyp\n')
            for c, rr, pp in zip(ids, r, p):
                f.write(f'{c}\t{rr}\t{pp}\n')
        json.dump({'step': step, 'bleu4': b.score, 'sacrebleu': str(b)}, open(os.path.join(a.run_dir, 'test_last.json'), 'w'), indent=1)
    P('DONE', a.run_dir)


if __name__ == '__main__':
    main()
