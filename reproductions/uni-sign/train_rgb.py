"""Uni-Sign pose + RGB training (--rgb_support), single GPU, stage 2 or stage 3 of the paper.

stage 2: start from the pose-only stage-1 checkpoint (pretrain.py, setting A). The RGB branch (ImageNet EfficientNet-B0,
         projection and fusion) is new and loaded with strict=False. YouTube-ASL caption segments, 5 epochs, 128 clips per
         update, lr 3e-4, cosine; dev BLEU after every epoch.
stage 3: start from the stage-2 checkpoint (strict) and fine-tune on one dataset, 20 epochs, 32 clips per update, lr 3e-4;
         the last epoch decodes the test split (beam 4, at most 100 new tokens) into test_predictions.tsv + test_scores.json.
Optimiser, schedule, bf16 autocast and gradient clip 1.0 as in pretrain.py / finetune.py.

Model input is built by Uni-Sign's own code: frame sampling as S2T_Dataset (at most --max_length frames, random.sample,
kept in order), load_part_kp for the pose, load_support_rgb_dict for the hand crops (score-aware choice of hand frames,
112 x 112 crops, ImageNet normalisation, frames read with decord). Our keypoints are on every second video frame, so the
pose frame chosen by Uni-Sign is mapped to its video frame (../common/videos.py).

Resumable: <out_dir_ft>/last.pth every --save_every updates and every epoch. A resumed epoch draws a fresh permutation and
skips the samples already used in it. --resume_from starts from the full state of another run when this run has no last.pth.
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

import unisign_setup as U
import datasets as US  # upstream Uni-Sign datasets.py
from videos import clip_video, frame_indices


class RGBPoseDataset(U.Base_Dataset):
    def __init__(self, data_root, split, index, video_root, args):
        super().__init__()
        self.args = args
        self.rgb_support = True
        self.max_length = args.max_length
        self.index = index
        self.video_root = video_root
        cap = U.load_captions(data_root, 'setting_a')
        ids = U.read_split(data_root, split)
        seen = {}

        def has_video(c):
            path = clip_video(c, video_root)[0]
            if path not in seen:
                seen[path] = os.path.exists(path)
            return seen[path]

        self.items = [(c, cap[c].strip()) for c in ids if c in cap and c in index and has_video(c)]
        self.n_missing = len(ids) - len(self.items)
        self.data_transform = US.transforms.Compose([US.transforms.ToTensor(),
                                                     US.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip, text = self.items[i]
        z = self.index.load(clip)
        K, S = z['keypoints'], z['scores']
        dur = len(S)
        video, fidx = frame_indices(clip, self.video_root, dur)
        tmp = sorted(random.sample(range(dur), k=self.max_length)) if dur > self.max_length else list(range(dur))
        tmp = np.array(tmp)
        sk = [np.asarray(K[j], np.float32)[None] for j in tmp]
        cf = [np.asarray(S[j], np.float32)[None] for j in tmp]
        kp_feats = {k: v.float() for k, v in U.load_part_kp(sk, cf, force_ok=True).items()}
        support = US.load_support_rgb_dict(np.asarray(fidx)[tmp], sk, cf, video, self.data_transform)
        return clip, kp_feats, text, '', support


def main():
    p = U.get_parser('Uni-Sign pose + RGB')
    p.add_argument('--stage', type=int, required=True, choices=[2, 3])
    p.add_argument('--init', required=True)
    p.add_argument('--out_dir_ft', required=True)
    p.add_argument('--video_root', required=True, help='source videos, layout in ../common/videos.py')
    p.add_argument('--train_split', required=True)
    p.add_argument('--dev_split', default='')
    p.add_argument('--test_split', default='')
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--resume_from', default='', help='full-state checkpoint of another run, used when this run has no last.pth')
    a = p.parse_args()
    a.rgb_support = True
    for k in ('data_root', 'out_dir_ft', 'cache_dir', 'init', 'video_root', 'resume_from'):
        setattr(a, k, U.absolute(getattr(a, k)))
    U.utils.set_seed(a.seed)
    os.makedirs(a.out_dir_ft, exist_ok=True)
    log = open(os.path.join(a.out_dir_ft, 'train.log'), 'a')

    def P(*s):
        m = ' '.join(str(x) for x in s)
        print(m, flush=True)
        log.write(m + '\n')
        log.flush()

    index = U.rtmpose_index(a.data_root, 'setting_a', a.cache_dir or None)
    tr = RGBPoseDataset(a.data_root, a.train_split, index, a.video_root, a)
    dv = RGBPoseDataset(a.data_root, a.dev_split, index, a.video_root, a) if a.dev_split else None
    te = RGBPoseDataset(a.data_root, a.test_split, index, a.video_root, a) if a.test_split else None
    P(f'stage {a.stage} | train {len(tr)} (missing {tr.n_missing}) dev {len(dv) if dv else 0} test {len(te) if te else 0} | '
      f'batch {a.batch_size} x accum {a.accum} = {a.batch_size * a.accum} | lr {a.lr} | epochs {a.epochs}')
    dl = DataLoader(tr, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=tr.collate_fn, shuffle=True, drop_last=True,
                    persistent_workers=a.num_workers > 0)
    model = U.Uni_Sign(args=a)
    dev = torch.device('cuda')
    sd = torch.load(a.init, map_location='cpu', weights_only=False)['model']
    ret = model.load_state_dict(sd, strict=(a.stage == 3))
    P(f'init {a.init} | missing {len(ret.missing_keys)} keys | unexpected {len(ret.unexpected_keys)}')
    if a.stage == 2:
        assert all(k.startswith(('rgb_', 'fusion_')) for k in ret.missing_keys) and not ret.unexpected_keys, 'only the RGB branch may be new'
    model.to(dev).train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    total = max(1, a.epochs * len(dl) // a.accum)
    sch = get_scheduler('cosine', optimizer=opt, num_warmup_steps=int(a.warmup_epochs * len(dl) / a.accum), num_training_steps=total)
    tok = model.mt5_tokenizer
    last = os.path.join(a.out_dir_ft, 'last.pth')
    ep0, step, skip = 0, 0, 0
    src_ck = last if os.path.exists(last) else a.resume_from
    if src_ck:
        ck = torch.load(src_ck, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sch.load_state_dict(ck['sch'])
        ep0, step, skip = ck['epoch'], ck['step'], ck['batch_in_epoch']
        P(f'resumed from {src_ck}: epoch {ep0}, update {step}; that epoch continues after {skip * a.batch_size * a.accum} samples')

    def save_last(ep, nb):
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'epoch': ep, 'step': step,
                    'batch_in_epoch': nb}, last + '.tmp')
        os.replace(last + '.tmp', last)

    def decode(ds):
        model.eval()
        names, preds, refs = [], [], []
        dlx = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=ds.collate_fn, shuffle=False)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for src, tgt in dlx:
                names += list(src['name_batch'])
                so = model(U.to_device(src, dev), tgt)
                out = model.generate(so, max_new_tokens=100, num_beams=4)
                preds += tok.batch_decode(out, skip_special_tokens=True)
                refs += list(tgt['gt_sentence'])
        model.train()
        b, r = U.translation_performance(refs, preds)
        return b, r, names, preds, refs

    for ep in range(ep0, a.epochs):
        run = []
        dlx = dl
        if ep == ep0 and skip:
            perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(a.seed + ep)).tolist()[skip * a.batch_size * a.accum:]
            dlx = DataLoader(torch.utils.data.Subset(tr, perm), batch_size=a.batch_size, num_workers=a.num_workers,
                             collate_fn=tr.collate_fn, shuffle=False, drop_last=True)
        off = skip * a.accum if (ep == ep0 and skip) else 0   # (i + 1) // accum stays the position in the full epoch
        for j, (src, tgt) in enumerate(dlx):
            i = j + off
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
                if step % 20 == 0:
                    P(f'ep {ep} step {step}/{total} loss {np.mean(run[-20 * a.accum:]):.4f} lr {sch.get_last_lr()[0]:.2e}')
                if step % a.save_every == 0:
                    save_last(ep, (i + 1) // a.accum)
        torch.save({'model': model.state_dict()}, os.path.join(a.out_dir_ft, f'checkpoint_ep{ep}.pth'))
        save_last(ep + 1, 0)
        P(f'saved checkpoint_ep{ep}.pth')
        if dv is not None:
            b, r, _, ps, rs = decode(dv)
            P(f'DEV ep {ep} ' + json.dumps({k: round(v, 2) for k, v in b.items()}) + f' rougeL {r:.2f}')
            for x, y in zip(rs[:3], ps[:3]):
                P(f'  ref: {x}\n  hyp: {y}')
    torch.save({'model': model.state_dict()}, os.path.join(a.out_dir_ft, 'final_checkpoint.pth'))
    P('saved final_checkpoint.pth (last epoch)')
    if te is not None:
        b, r, names, ps, rs = decode(te)
        with open(os.path.join(a.out_dir_ft, 'test_predictions.tsv'), 'w') as f:
            f.write('clip_id\tref\thyp\n')
            for n, x, y in zip(names, rs, ps):
                f.write(f'{n}\t{x}\t{y}\n'.replace('\r', ' '))
        json.dump({'bleu': b, 'rougeL': r, 'n': len(ps)}, open(os.path.join(a.out_dir_ft, 'test_scores.json'), 'w'), indent=1)
        P('TEST ' + json.dumps({k: round(v, 2) for k, v in b.items()}) + f' rougeL {r:.2f} (n {len(ps)})')
    P('DONE', a.out_dir_ft)


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
