"""Score a Uni-Sign pose-only translation checkpoint on a test split, without deepspeed.

Decoding as the upstream fine_tuning.evaluate: model in bf16, beam 4, at most 100 new tokens, at most 256 frames per clip.
BLEU from the upstream SLRT_metrics.translation_performance (sacreBLEU, 13a). Writes <out_dir_eval>/metrics.json,
preds.txt, refs.txt and test_predictions.tsv (clip_id / ref / hyp, read by ../common/score_openasl_split.py).
"""
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import unisign_setup as U


def main():
    p = U.get_parser('Uni-Sign evaluation')
    p.add_argument('--setting', default='setting_a', choices=['setting_a', 'setting_b'], help='which keypoint folders and captions')
    p.add_argument('--split', required=True, help='e.g. setting_a/test_how2sign')
    p.add_argument('--out_dir_eval', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    a.rgb_support = False
    a.data_root, a.out_dir_eval, a.cache_dir, a.finetune = (U.absolute(x) for x in (a.data_root, a.out_dir_eval, a.cache_dir, a.finetune))
    U.utils.set_seed(a.seed)
    os.makedirs(a.out_dir_eval, exist_ok=True)
    index = U.rtmpose_index(a.data_root, a.setting, a.cache_dir or None)
    ds = U.LabelPoseDataset(U.translation_items(a.data_root, a.setting, a.split), index, a)
    print(f'{len(ds)} clips with keypoints ({ds.n_missing} without) from {a.split}', flush=True)
    dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=ds.collate_fn, shuffle=False)
    model = U.Uni_Sign(args=a)
    ret = model.load_state_dict(torch.load(a.finetune, map_location='cpu')['model'], strict=True)
    print('missing', ret.missing_keys, 'unexpected', ret.unexpected_keys, flush=True)
    dev = torch.device(a.device)
    model.to(dev).eval().to(torch.bfloat16)
    tok = model.mt5_tokenizer
    names, preds, refs, losses = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for step, (src, tgt) in enumerate(dl):
            for k in src:
                if isinstance(src[k], torch.Tensor):
                    src[k] = src[k].to(torch.bfloat16).to(dev)
            so = model(src, tgt)
            losses.append(float(so['loss']))
            out = model.generate(so, max_new_tokens=100, num_beams=4)
            preds += tok.batch_decode(out, skip_special_tokens=True)
            refs += list(tgt['gt_sentence'])
            names += list(src['name_batch'])
            if step % 10 == 0:
                print(f'  [{step}/{len(dl)}] {(time.time() - t0) / 60:.1f} min', flush=True)
    bleu, rouge = U.translation_performance(refs, preds)
    m = {'split': a.split, 'ckpt': a.finetune, 'n': len(preds), 'n_missing_keypoints': ds.n_missing,
         **{k: round(v, 2) for k, v in bleu.items()}, 'rouge_l': round(rouge, 2), 'loss': round(float(np.mean(losses)), 4)}
    print(json.dumps(m, indent=1), flush=True)
    json.dump(m, open(os.path.join(a.out_dir_eval, 'metrics.json'), 'w'), indent=1)
    with open(os.path.join(a.out_dir_eval, 'preds.txt'), 'w') as f:
        for n_, p_ in zip(names, preds):
            f.write(f'sample: {n_}, prediction: {p_}\n')
    with open(os.path.join(a.out_dir_eval, 'refs.txt'), 'w') as f:
        for n_, r_ in zip(names, refs):
            f.write(f'sample: {n_}, ground-truth: {r_}\n')
    with open(os.path.join(a.out_dir_eval, 'test_predictions.tsv'), 'w') as f:
        f.write('clip_id\tref\thyp\n')
        for n_, r_, p_ in zip(names, refs, preds):
            f.write(f'{n_}\t{r_}\t{p_}\n')


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
