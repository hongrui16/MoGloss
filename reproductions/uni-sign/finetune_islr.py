"""Uni-Sign isolated sign recognition on WLASL, single GPU, without deepspeed.

The released WLASL recipe of Uni-Sign (--task ISLR, script/train_stage3.sh of the upstream repository):
  - the target text is the gloss, generated with beam 4 (at most 100 new tokens); a prediction is correct when the decoded
    string equals the gloss (SLRT_metrics.islr_performance, per-instance and per-class top-1);
  - 20 epochs, lr 3e-4, AdamW weight decay 1e-4, cosine, 32 clips per update (here --batch-size 8 --accum 4 for 8 x 4 GPUs),
    at most --max_length 64 frames per clip;
  - dev after every epoch, best_checkpoint.pth = best dev per-instance top-1; the test split is scored once with that
    checkpoint (test_bestdev.json + predictions). final_checkpoint.pth = last epoch.
Input: keypoints/rtmpose_2d/wlasl2000_annotation_cut (the files cut to the annotated span), splits/islr/wlasl{100,2000}_*,
labels = the gloss in metadata/wlasl2000.jsonl. Clips without keypoints are skipped (one WLASL2000 training clip whose
annotated span lies outside its video).
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

import unisign_setup as U
from release_data import ShardIndex


def main():
    p = U.get_parser('Uni-Sign ISLR fine-tuning')
    p.add_argument('--subset', required=True, choices=['wlasl100', 'wlasl2000'])
    p.add_argument('--out_dir_ft', required=True)
    p.add_argument('--accum', type=int, default=4)
    a = p.parse_args()
    a.rgb_support = False
    a.task = 'ISLR'
    a.data_root, a.out_dir_ft, a.cache_dir, a.finetune = (U.absolute(x) for x in (a.data_root, a.out_dir_ft, a.cache_dir, a.finetune))
    U.utils.set_seed(a.seed)
    os.makedirs(a.out_dir_ft, exist_ok=True)
    log = open(os.path.join(a.out_dir_ft, 'train.log'), 'a')

    def P(*s):
        m = ' '.join(str(x) for x in s)
        print(m, flush=True)
        log.write(m + '\n')
        log.flush()

    index = ShardIndex(a.data_root, ['keypoints/rtmpose_2d/wlasl2000_annotation_cut'], a.cache_dir or None)
    meta = U.load_metadata(a.data_root, 'wlasl2000')

    def items(split):
        return [(c, meta[c]['gloss'], [meta[c]['gloss']]) for c in U.read_split(a.data_root, f'islr/{a.subset}_{split}')]

    tr, dv, te = (U.LabelPoseDataset(items(s), index, a) for s in ('train', 'val', 'test'))
    P(f'train {len(tr)} dev {len(dv)} test {len(te)} | missing keypoints {tr.n_missing}/{dv.n_missing}/{te.n_missing} | '
      f'batch {a.batch_size} x accum {a.accum} | lr {a.lr} | max_length {a.max_length}')
    dl = DataLoader(tr, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=tr.collate_fn, shuffle=True, drop_last=True)

    def mk(ds):
        return DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=ds.collate_fn, shuffle=False)

    dd, dt = mk(dv), mk(te)
    model = U.Uni_Sign(args=a)
    ret = model.load_state_dict(torch.load(a.finetune, map_location='cpu')['model'], strict=True)
    P('loaded', a.finetune, ret)
    dev = torch.device('cuda')
    model.to(dev).train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    total = max(1, a.epochs * len(dl) // a.accum)
    sch = get_scheduler('cosine', optimizer=opt, num_warmup_steps=int(a.warmup_epochs * len(dl) / a.accum), num_training_steps=total)
    tok = model.mt5_tokenizer
    best, hist = -1, []

    def evaluate(loader):
        model.eval()
        names, preds, refs = [], [], []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for src, tgt in loader:
                names += list(src['name_batch'])
                so = model(U.to_device(src, dev), tgt)
                out = model.generate(so, max_new_tokens=100, num_beams=4)
                preds += tok.batch_decode(out, skip_special_tokens=True)
                refs += list(tgt['gt_sentence'])
        model.train()
        pi, pc = U.islr_performance(refs, preds)
        return pi, pc, names, preds, refs

    step = 0
    for ep in range(a.epochs):
        run = []
        for i, (src, tgt) in enumerate(dl):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(U.to_device(src, dev), tgt)['loss'] / a.accum
            loss.backward()
            run.append(loss.item() * a.accum)
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    P(f'ep {ep} step {step}/{total} loss {np.mean(run[-50 * a.accum:]):.4f} lr {sch.get_last_lr()[0]:.2e}')
        pi, pc, _, ps, rs = evaluate(dd)
        hist.append({'epoch': ep, 'dev_pi': pi, 'dev_pc': pc})
        P(f'DEV ep {ep} top1 P-I {pi:.2f} P-C {pc:.2f} | e.g. ' + ' ; '.join(f'{r} -> {h}' for r, h in zip(rs[:3], ps[:3])))
        if pi > best:
            best = pi
            torch.save({'model': model.state_dict(), 'epoch': ep}, os.path.join(a.out_dir_ft, 'best_checkpoint.pth'))
            P('  -> new best')
    torch.save({'model': model.state_dict()}, os.path.join(a.out_dir_ft, 'final_checkpoint.pth'))
    ck = torch.load(os.path.join(a.out_dir_ft, 'best_checkpoint.pth'), map_location='cpu')
    model.load_state_dict(ck['model'])
    pi, pc, ns, ps, rs = evaluate(dt)
    res = {'init': a.finetune, 'best_dev_epoch': ck['epoch'], 'best_dev_pi': best, 'test_pi': pi, 'test_pc': pc, 'n_test': len(ps),
           'dev_history': hist}
    json.dump(res, open(os.path.join(a.out_dir_ft, 'test_bestdev.json'), 'w'), indent=1)
    with open(os.path.join(a.out_dir_ft, 'test_bestdev_predictions.tsv'), 'w') as f:
        for n, r, h in zip(ns, rs, ps):
            f.write(f'{n}\t{r}\t{h}\n')
    P('TEST ' + json.dumps({k: v for k, v in res.items() if k != 'dev_history'}))
    P('DONE', a.out_dir_ft)


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
