"""BLEU-1..4 of OpenASL test predictions on all 908 sentences and on the sentences from videos unseen / seen in YouTube-ASL.

Input: a tsv with columns clip_id / ref / hyp (clip_id = released clip name). sacreBLEU, 13a tokenization, case-sensitive.
  python score_openasl_split.py <test_predictions.tsv> --data_root <release> [--out scores.json]
"""
import argparse
import csv
import json
import os
import sys

from sacrebleu.metrics import BLEU

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from release_data import read_split  # noqa: E402

csv.field_size_limit(10 ** 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pred')
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    data = {r['clip_id']: (r['ref'], r['hyp']) for r in csv.DictReader(open(a.pred), delimiter='\t', quoting=csv.QUOTE_NONE)}
    out = {'pred': a.pred}
    for part, split in (('all', 'setting_a/test_openasl'), ('unseen', 'openasl_test_unseen'), ('seen', 'openasl_test_seen')):
        ids = read_split(a.data_root, split)
        miss = [i for i in ids if i not in data]
        r = [data[i][0] for i in ids if i in data]
        h = [data[i][1] for i in ids if i in data]
        out[part] = {'n': len(r), 'missing': len(miss),
                     'bleu_1to4': [round(BLEU(max_ngram_order=n).corpus_score(h, [r]).score, 2) for n in (1, 2, 3, 4)]}
    print(json.dumps(out))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
