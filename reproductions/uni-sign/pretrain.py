"""Uni-Sign pose-only translation training from mT5-Base and randomly initialised pose encoders.

Used for two runs of the paper:
  setting A, stage 1: YouTube-ASL caption segments (setting_a/pretrain_youtube_asl_train), dev = setting_a/pretrain_youtube_asl_dev.
  setting B: all data (setting_b/train), dev = setting_b/val without its STEM clips, then the test splits are decoded with the
             last epoch and with the best dev epoch.
The pretraining script of Uni-Sign is not released. Its objective is the same sequence-to-sequence loss as fine-tuning, so this
is the fine-tuning loop on the pretraining data with the recipe of the paper: AdamW (upstream defaults, weight decay 1e-4),
lr 3e-4, cosine, 20 epochs, 512 clips per update (here --batch-size 16 --accum 32 on one GPU), bf16 autocast, gradient clip 1.0,
at most 256 frames per clip.
Resumable: <out_dir>/last.pth after every epoch (--resume). checkpoint_ep<k>.pth every epoch, final_checkpoint.pth at the end.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

import unisign_setup as U


def main():
    p = U.get_parser('Uni-Sign pose-only translation training')
    p.add_argument('--setting', required=True, choices=['setting_a', 'setting_b'])
    p.add_argument('--train_split', required=True)
    p.add_argument('--dev_split', required=True)
    p.add_argument('--dev_exclude_prefix', default='', help="drop dev clips whose name starts with this prefix (setting B: 'stem__')")
    p.add_argument('--test_splits', default='', help='comma list of splits decoded after training (setting B)')
    p.add_argument('--out_dir_ft', required=True)
    p.add_argument('--accum', type=int, default=32)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    a.rgb_support = False
    a.data_root, a.out_dir_ft, a.cache_dir = U.absolute(a.data_root), U.absolute(a.out_dir_ft), U.absolute(a.cache_dir)
    U.utils.set_seed(a.seed)
    os.makedirs(a.out_dir_ft, exist_ok=True)
    log = open(os.path.join(a.out_dir_ft, 'train.log'), 'a')

    def P(*s):
        msg = ' '.join(str(x) for x in s)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    index = U.rtmpose_index(a.data_root, a.setting, a.cache_dir or None)
    dev_items = [it for it in U.translation_items(a.data_root, a.setting, a.dev_split)
                 if not (a.dev_exclude_prefix and it[0].startswith(a.dev_exclude_prefix))]
    tr = U.SegmentPoseDataset(U.translation_items(a.data_root, a.setting, a.train_split), index, a)
    dv = U.SegmentPoseDataset(dev_items, index, a)
    P(f'train {len(tr)} clips ({tr.n_missing} without keypoints), dev {len(dv)} | batch {a.batch_size} x accum {a.accum} = {a.batch_size * a.accum}')
    dl = DataLoader(tr, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=tr.collate_fn, shuffle=True, drop_last=True,
                    persistent_workers=a.num_workers > 0)
    dd = DataLoader(dv, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=dv.collate_fn, shuffle=False)
    model = U.Uni_Sign(args=a)
    dev = torch.device('cuda')
    model.to(dev).train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    total = max(1, a.epochs * len(dl) // a.accum)
    sch = get_scheduler('cosine', optimizer=opt, num_warmup_steps=int(a.warmup_epochs * len(dl) / a.accum), num_training_steps=total)
    ep0, step = 0, 0
    last = os.path.join(a.out_dir_ft, 'last.pth')
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sch.load_state_dict(ck['sch'])
        ep0, step = ck['epoch'] + 1, ck['step']
        P(f'resumed from {last}: next epoch {ep0}, step {step}')
    else:
        P('start: mT5-base + random pose encoders')
    tok = model.mt5_tokenizer

    def decode(loader):
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
        return names, refs, preds

    def run_tests(tag):
        from sacrebleu.metrics import BLEU
        res = {}
        for split in [s for s in a.test_splits.split(',') if s]:
            ds = U.SegmentPoseDataset(U.translation_items(a.data_root, a.setting, split), index, a)
            dlt = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=ds.collate_fn, shuffle=False)
            names, refs, preds = decode(dlt)
            b14 = [round(BLEU(max_ngram_order=n).corpus_score(preds, [refs]).score, 2) for n in (1, 2, 3, 4)]
            res[split] = {'n': len(preds), 'bleu_1to4': b14}
            P(f'TEST {tag} {split} BLEU-4 {b14[3]} | n {len(preds)}')
            with open(os.path.join(a.out_dir_ft, f'test_{tag}_{split.replace("/", "_")}_predictions.tsv'), 'w') as f:
                f.write('clip_id\tref\thyp\n')
                for n_, r_, p_ in zip(names, refs, preds):
                    f.write(f'{n_}\t{r_}\t{p_}\n')
        json.dump(res, open(os.path.join(a.out_dir_ft, f'test_{tag}.json'), 'w'), indent=1)

    hist_path = os.path.join(a.out_dir_ft, 'dev_history.json')
    for ep in range(ep0, a.epochs):
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
        _, refs, preds = decode(dd)
        b, r = U.translation_performance(refs, preds)
        P(f'DEV ep {ep} ' + json.dumps({k: round(v, 2) for k, v in b.items()}) + f' rougeL {r:.2f}')
        hist = json.load(open(hist_path)) if os.path.exists(hist_path) else {}
        hist[str(ep)] = b['bleu4']
        json.dump(hist, open(hist_path, 'w'))
        for x, y in zip(refs[:3], preds[:3]):
            P(f'  ref: {x}\n  hyp: {y}')
        torch.save({'model': model.state_dict()}, os.path.join(a.out_dir_ft, f'checkpoint_ep{ep}.pth'))
        tmp = last + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'epoch': ep, 'step': step}, tmp)
        os.replace(tmp, last)
        P(f'saved epoch {ep}')
    torch.save({'model': model.state_dict()}, os.path.join(a.out_dir_ft, 'final_checkpoint.pth'))
    if a.test_splits:
        run_tests('last')
        hist = json.load(open(hist_path))
        be = max(hist, key=lambda k: hist[k])
        P(f'best dev epoch {be} (dev BLEU-4 {hist[be]:.2f})')
        ck = torch.load(os.path.join(a.out_dir_ft, f'checkpoint_ep{be}.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        run_tests(f'bestdev_ep{be}')
    P('DONE', a.out_dir_ft)


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
