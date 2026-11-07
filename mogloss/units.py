"""Pseudo-gloss vocabulary: special tokens, the most frequent training units, then every lexicon entry not yet included."""
from collections import Counter

from .models import MAX_UNITS

SPECIALS = ['<pad>', '<mask>', '<unk>']


def build_vocab(unit_lists, lexicon=(), size=30000):
    c = Counter()
    for u in unit_lists:
        c.update(u)
    voc = SPECIALS + [w for w, _ in c.most_common(size)]
    seen = set(voc)
    voc += [w for w in lexicon if w not in seen]
    return {w: i for i, w in enumerate(voc)}


def read_lexicon(path):
    """First column of a tab-separated lexicon file with a header row."""
    return [l.split('\t')[0] for l in open(path).read().splitlines()[1:] if l.strip()]


def encode_units(units, voc):
    unk = voc['<unk>']
    return [voc.get(u, unk) for u in units[:MAX_UNITS]]
