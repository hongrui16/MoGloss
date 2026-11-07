"""Reading the released data (https://huggingface.co/datasets/HenryH101/MoGloss).

root/
  keypoints/rtmpose_2d/<dataset>/<dataset>-NNNNN.tar   one <clip>.npz per clip (keypoints (T,133,2), scores (T,133))
  metadata/<dataset>.jsonl                             captions and pseudo-glosses per setting
  splits/...                                           lists of clip names
"""
import glob
import io
import json
import os
import tarfile

import numpy as np

from .features import frame_features

# Setting A reads these clips from the box-guided OpenASL extraction (same names as in keypoints/rtmpose_2d/openasl).
SETTING_A_OVERRIDE = 'openasl_setting_a_box'


class ClipStore:
    """Random access to the clips of the tar shards. An index (clip -> shard, offset, size) is built once per dataset
    folder and cached as index.json next to the shards (or in cache_dir, or in $MOGLOSS_CACHE)."""

    def __init__(self, root, setting='b', datasets=None, cache_dir=None):
        self.root = os.path.join(root, 'keypoints', 'rtmpose_2d')
        self.cache_dir = cache_dir or os.environ.get('MOGLOSS_CACHE')
        names = datasets or sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)))
        self.loc = {}
        for name in names:
            if name == SETTING_A_OVERRIDE:
                continue
            self.loc.update(self._index(name))
        if setting == 'a' and os.path.isdir(os.path.join(self.root, SETTING_A_OVERRIDE)):
            self.loc.update(self._index(SETTING_A_OVERRIDE))

    def _index(self, name):
        d = os.path.join(self.root, name)
        cache = os.path.join(self.cache_dir, f'{name}_index.json') if self.cache_dir else os.path.join(d, 'index.json')
        if os.path.exists(cache):
            idx = json.load(open(cache))
        else:
            idx = {}
            for t in sorted(glob.glob(os.path.join(d, '*.tar'))):
                with tarfile.open(t) as tf:
                    for m in tf.getmembers():
                        idx[m.name[:-4]] = (os.path.basename(t), m.offset_data, m.size)
            try:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                json.dump(idx, open(cache, 'w'))
            except OSError:
                pass
        return {k: (os.path.join(d, v[0]), v[1], v[2]) for k, v in idx.items()}

    def __contains__(self, clip):
        return clip in self.loc

    def raw(self, clip):
        path, off, size = self.loc[clip]
        with open(path, 'rb') as f:
            f.seek(off)
            z = np.load(io.BytesIO(f.read(size)))
            return z['keypoints'], z['scores']

    def features(self, clip):
        """(T, 207) frame features of a clip."""
        return frame_features(*self.raw(clip))


def read_list(path):
    return [l.strip() for l in open(path) if l.strip()]


def load_metadata(root, datasets=None):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, 'metadata', '*.jsonl'))):
        name = os.path.basename(f)[:-6]
        if datasets and name not in datasets:
            continue
        for line in open(f):
            r = json.loads(line)
            out[r['id']] = r
    return out


def caption(meta, clip, setting):
    r = meta[clip].get(f'setting_{setting}')
    return r['caption'].strip() if r and r.get('caption') else ''


def pseudo_gloss(meta, clip, setting):
    r = meta[clip].get(f'setting_{setting}')
    return r.get('pseudo_gloss') if r else None
