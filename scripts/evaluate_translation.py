"""Evaluate a translation checkpoint on one or more test lists (beam 4, sacreBLEU BLEU-4, ROUGE-L).

  python scripts/evaluate_translation.py --data_root <release> --setting b \
      --ckpt <release>/weights/mogloss/setting_b/translation/mmm_joint/all_data.pt \
      --test_lists splits/setting_b/test_stem.txt splits/setting_b/test_openasl.txt \
                   splits/setting_b/test_how2sign.txt splits/setting_b/test_youtube_asl.txt
For an OpenASL test list the scores on the seen and unseen videos (splits/openasl_test_{seen,unseen}.txt) are added.
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mogloss.data import ClipStore, caption, load_metadata, read_list
from mogloss.models import Translator
from mogloss.translation import score, translate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--setting', choices=['a', 'b'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--test_lists', nargs='+', required=True)
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--batch_size', type=int, default=64)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rel = lambda p: p if os.path.isabs(p) else os.path.join(a.data_root, p)
    store = ClipStore(a.data_root, a.setting)
    meta = load_metadata(a.data_root)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    c = ck['config']
    model = Translator(c.get('din', 207), c['d'], c['layers'], c['heads'], c['max_len'])
    model.load_state_dict(ck['model'], strict=True)
    model = model.to(dev)
    tok = AutoTokenizer.from_pretrained('t5-base')
    seen = set(read_list(os.path.join(a.data_root, 'splits', 'openasl_test_seen.txt')))
    unseen = set(read_list(os.path.join(a.data_root, 'splits', 'openasl_test_unseen.txt')))
    results = {}
    for lst in a.test_lists:
        items = [(k, caption(meta, k, a.setting)) for k in read_list(rel(lst)) if k in store and caption(meta, k, a.setting)]
        p, r, clips = translate(model, store, items, tok, dev, a.batch_size)
        name = os.path.splitext(os.path.basename(lst))[0]
        res = {'all': score(p, r)}
        if any(k in seen for k in clips):
            for part, ids in (('seen', seen), ('unseen', unseen)):
                sel = [i for i, k in enumerate(clips) if k in ids]
                res[part] = score([p[i] for i in sel], [r[i] for i in sel])
        results[name] = res
        print(name, json.dumps({k: round(v['bleu4'], 2) for k, v in res.items()}), flush=True)
        if a.out_dir:
            os.makedirs(a.out_dir, exist_ok=True)
            with open(os.path.join(a.out_dir, f'{name}_predictions.tsv'), 'w') as f:
                f.write('clip\tref\thyp\n')
                for k, rr, pp in zip(clips, r, p):
                    f.write(f'{k}\t{rr}\t{pp}\n')
    if a.out_dir:
        json.dump(results, open(os.path.join(a.out_dir, 'scores.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
