"""Uni-Sign pose-only fine-tuning on one dataset (setting A, stage 3), single GPU, without deepspeed.

Same loop as the upstream fine_tuning.py with the paper recipe: AdamW (upstream defaults, weight decay 1e-4), lr 3e-4, cosine,
20 epochs, 32 clips per update (here --batch-size 8 --accum 4 for the paper's 8 x 4 GPUs), bf16 autocast, gradient clip 1.0,
at most 256 frames per clip. There is no dev set and no checkpoint selection: the last epoch is kept (final_checkpoint.pth)
and scored on the test split afterwards with evaluate.py.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

import unisign_setup as U


def main():
    p = U.get_parser('Uni-Sign fine-tuning')
    p.add_argument('--train_split', required=True, help='e.g. setting_a/finetune_how2sign_train')
    p.add_argument('--out_dir_ft', required=True)
    p.add_argument('--accum', type=int, default=4)
    a = p.parse_args()
    a.rgb_support = False
    a.data_root, a.out_dir_ft, a.cache_dir, a.finetune = (U.absolute(x) for x in (a.data_root, a.out_dir_ft, a.cache_dir, a.finetune))
    U.utils.set_seed(a.seed)
    os.makedirs(a.out_dir_ft, exist_ok=True)
    log = open(os.path.join(a.out_dir_ft, 'train.log'), 'a')

    def P(*s):
        m = ' '.join(str(x) for x in s)
        print(m, flush=True)
        log.write(m + '\n')
        log.flush()

    index = U.rtmpose_index(a.data_root, 'setting_a', a.cache_dir or None)
    tr = U.LabelPoseDataset(U.translation_items(a.data_root, 'setting_a', a.train_split), index, a)
    P(f'train {len(tr)} ({tr.n_missing} without keypoints) | batch {a.batch_size} x accum {a.accum} | lr {a.lr}')
    dl = DataLoader(tr, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=tr.collate_fn, shuffle=True, drop_last=True)
    model = U.Uni_Sign(args=a)
    ret = model.load_state_dict(torch.load(a.finetune, map_location='cpu')['model'], strict=True)
    P('loaded', a.finetune, ret)
    dev = torch.device('cuda')
    model.to(dev).train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    total = max(1, a.epochs * len(dl) // a.accum)
    sch = get_scheduler('cosine', optimizer=opt, num_warmup_steps=int(a.warmup_epochs * len(dl) / a.accum), num_training_steps=total)

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
        torch.save({'model': model.state_dict()}, os.path.join(a.out_dir_ft, f'checkpoint_ep{ep}.pth'))
        P(f'saved checkpoint_ep{ep}.pth')
    torch.save({'model': model.state_dict()}, os.path.join(a.out_dir_ft, 'final_checkpoint.pth'))
    P('DONE', a.out_dir_ft)


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
