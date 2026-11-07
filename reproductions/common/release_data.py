"""Read the MoGloss Hugging Face release: keypoint tar shards, metadata and split lists.

Layout under --data_root (the downloaded dataset repo):
  keypoints/rtmpose_2d/<folder>/<folder>-NNNNN.tar   one <clip>.npz per clip (keypoints (T,133,2), scores (T,133))
  keypoints/mediapipe/mediapipe-NNNNN.tar            one <clip>.npz per clip (MediaPipe landmarks)
  metadata/<corpus>.jsonl                            one JSON line per clip
  splits/setting_a/*.txt, splits/setting_b/*.txt, splits/islr/*.txt

Clips are read in place from the tar files. The byte offset of every member is indexed once and cached.
"""
import glob
import io
import json
import os
import re
import tarfile

import numpy as np

# Keypoint folders read by each setting. When a clip is in several folders, the first one wins.
# Setting A reads the box-guided re-extraction of the multi-person OpenASL clips.
FOLDERS = {
    'setting_a': ['youtube_asl_raw_captions', 'how2sign', 'asl_stem_wiki', 'openasl_setting_a_box', 'openasl'],
    'setting_b': ['youtube_asl', 'youtube_sl25_ase', 'how2sign', 'asl_stem_wiki', 'openasl'],
}
CONTINUOUS = ['youtube_asl', 'youtube_asl_raw_captions', 'youtube_sl25_ase', 'how2sign', 'asl_stem_wiki', 'openasl']


def internal_id(clip):
    """Clip name with ':' restored as in our internal ids (used only to reproduce the clip order of our runs)."""
    m = re.match(r'oa__(.+)-(\d+__\d+__[\d.]+)-(\d+__\d+__[\d.]+)$', clip)
    if m:
        return 'oa:' + m.group(1) + '-' + m.group(2).replace('__', ':') + '-' + m.group(3).replace('__', ':')
    for p in ('ytseg__', 'yt__', 'sl25__', 'h2s__', 'stem__'):
        if clip.startswith(p):
            return p[:-2] + ':' + clip[len(p):]
    return clip


def read_split(data_root, name):
    """Clip names of a split. `name` is a file path or a name under splits/ such as 'setting_a/test_how2sign'."""
    path = name if os.path.isfile(name) else os.path.join(data_root, 'splits', name + '.txt')
    return [l.strip() for l in open(path) if l.strip()]


def load_captions(data_root, setting):
    """{clip: caption} of one setting ('setting_a' or 'setting_b') over all continuous corpora."""
    out = {}
    for c in CONTINUOUS:
        p = os.path.join(data_root, 'metadata', f'{c}.jsonl')
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get(setting):
                out[r['id']] = r[setting]['caption']
    return out


def load_metadata(data_root, corpus):
    """{clip: record} of one metadata file (e.g. 'wlasl2000')."""
    return {r['id']: r for r in map(json.loads, open(os.path.join(data_root, 'metadata', f'{corpus}.jsonl')))}


class ShardIndex:
    """clip name -> (tar file, byte offset, size) over one or more keypoint folders.

    folders: paths relative to data_root, e.g. 'keypoints/rtmpose_2d/how2sign' or 'keypoints/mediapipe'.
    cache_dir: where the index of each folder is stored (built on first use).
    """

    def __init__(self, data_root, folders, cache_dir=None):
        self.loc = {}
        cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.cache', 'mogloss_index')
        for f in folders:
            for k, v in self._folder_index(os.path.join(data_root, f), cache_dir).items():
                self.loc.setdefault(k, v)

    @staticmethod
    def _folder_index(folder, cache_dir):
        tars = sorted(glob.glob(os.path.join(folder, '*.tar')))
        if not tars:
            raise FileNotFoundError(f'no tar shards in {folder}')
        key = os.path.abspath(folder).strip('/').replace('/', '_')
        cache = os.path.join(cache_dir, key + '.json')
        sig = [[os.path.basename(t), os.path.getsize(t)] for t in tars]
        if os.path.exists(cache):
            d = json.load(open(cache))
            if d['shards'] == sig:
                return {k: (os.path.join(folder, t), o, s) for k, (t, o, s) in d['clips'].items()}
        clips = {}
        for t in tars:
            with tarfile.open(t) as tf:
                for m in tf:
                    if m.isfile() and m.name.endswith('.npz'):
                        clips[m.name[:-4]] = (os.path.basename(t), m.offset_data, m.size)
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cache + f'.tmp{os.getpid()}'
        json.dump({'shards': sig, 'clips': clips}, open(tmp, 'w'))
        os.replace(tmp, cache)
        return {k: (os.path.join(folder, t), o, s) for k, (t, o, s) in clips.items()}

    def __contains__(self, clip):
        return clip in self.loc

    def __len__(self):
        return len(self.loc)

    def load(self, clip):
        """dict of arrays of one clip."""
        path, off, size = self.loc[clip]
        with open(path, 'rb') as f:
            f.seek(off)
            buf = f.read(size)
        z = np.load(io.BytesIO(buf))
        return {k: z[k] for k in z.files}


def rtmpose_index(data_root, setting, cache_dir=None, folders=None):
    """ShardIndex over the RTMPose folders of a setting (or an explicit list of folder names)."""
    names = folders or FOLDERS[setting]
    return ShardIndex(data_root, [os.path.join('keypoints', 'rtmpose_2d', n) for n in names], cache_dir)
