"""Data, decoding and scoring for sign-to-English translation."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

MAX_FRAMES = 1024  # longer clips are truncated


class ClipSet(Dataset):
    def __init__(self, store, items):
        """items: [(clip, caption)]"""
        self.store, self.items = store, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip, text = self.items[i]
        return torch.from_numpy(self.store.features(clip)[:MAX_FRAMES]), text, clip


def collate(b):
    L = max(len(x) for x, _, _ in b)
    x = torch.zeros(len(b), L, b[0][0].shape[1])
    m = torch.zeros(len(b), L, dtype=torch.bool)
    for i, (xx, _, _) in enumerate(b):
        x[i, :len(xx)], m[i, :len(xx)] = xx, True
    return x, m, [t for _, t, _ in b], [c for _, _, c in b]


def score(preds, refs):
    """sacreBLEU corpus BLEU-1..4 and ROUGE-L (F1, stemmed)."""
    import sacrebleu
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    out = {'n': len(refs), 'bleu4': sacrebleu.corpus_bleu(preds, [refs]).score,
           'rougeL': 100 * float(np.mean([rs.score(r, p)['rougeL'].fmeasure for r, p in zip(refs, preds)])) if refs else 0.0}
    out['bleu_1to4'] = [round(sacrebleu.metrics.BLEU(max_ngram_order=n).corpus_score(preds, [refs]).score, 2) for n in (1, 2, 3, 4)]
    return out


@torch.no_grad()
def translate(model, store, items, tok, dev, bs=64, workers=4):
    model.eval()
    preds, refs, clips = [], [], []
    for x, m, t, c in DataLoader(ClipSet(store, items), bs, shuffle=False, num_workers=workers, collate_fn=collate):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            preds += model.generate(x.to(dev), m.to(dev), tok)
        refs += t
        clips += c
    model.train()
    return preds, refs, clips


def micro_step(model, x, m, lab, dev, max_tokens):
    """loss.backward() for one batch. If B x L > max_tokens the batch is split into chunks whose losses are weighted by
    their share of target tokens, which gives the same gradient as the full batch. Returns the batch loss."""
    B, L = x.shape[0], x.shape[1]
    if not max_tokens or B * L <= max_tokens:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x.to(dev), m.to(dev), lab.to(dev))
        loss.backward()
        return loss.item()
    cb = max(1, max_tokens // L)
    n_all = (lab != -100).sum().item()
    tot = 0.0
    for i in range(0, B, cb):
        li = lab[i:i + cb]
        w = (li != -100).sum().item() / max(1, n_all)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x[i:i + cb].to(dev), m[i:i + cb].to(dev), li.to(dev))
        (loss * w).backward()
        tot += loss.item() * w
    return tot
