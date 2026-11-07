"""Load the official Uni-Sign code and read the MoGloss release for it.

The upstream repository (https://github.com/ZechengLi19/Uni-Sign) is used unmodified: model (Uni_Sign), pose features
(load_part_kp), RGB hand crops (load_support_rgb_dict), batching (Base_Dataset.collate_fn), argument defaults
(utils.get_args_parser) and metrics (SLRT_metrics). Pass its location with --unisign_repo (or the UNISIGN_REPO variable).

Its config.py reads mT5-Base from ./pretrained_weight/mt5-base, so the working directory is switched to the repository;
every path argument of our scripts is resolved against the directory the script was started from (see absolute()).
deepspeed and decord are imported by upstream modules but are not needed by the pose-only scripts; when they are not
installed they are replaced by empty modules. The RGB script needs a real decord.
"""
import argparse
import importlib.machinery
import os
import random
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
ORIG_CWD = os.getcwd()


def _repo_from_argv():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--unisign_repo', default=os.environ.get('UNISIGN_REPO', ''))
    a, _ = pre.parse_known_args()
    if not a.unisign_repo:
        sys.exit('pass --unisign_repo <checkout of https://github.com/ZechengLi19/Uni-Sign> or set UNISIGN_REPO')
    return os.path.abspath(a.unisign_repo)


REPO = _repo_from_argv()
sys.path.insert(0, REPO)
os.chdir(REPO)

for _m in ('deepspeed', 'deepspeed.comm', 'deepspeed.accelerator', 'decord'):
    try:
        __import__(_m)
    except ImportError:
        _stub = types.ModuleType(_m)
        _stub.__spec__ = importlib.machinery.ModuleSpec(_m, None)
        if _m == 'deepspeed.accelerator':
            _stub.get_accelerator = lambda: None
        if _m == 'decord':
            _stub.VideoReader = None
            _stub.cpu = None
        sys.modules[_m] = _stub
        if '.' in _m:
            setattr(sys.modules[_m.split('.')[0]], _m.split('.')[1], _stub)

import utils  # noqa: E402
from models import Uni_Sign  # noqa: E402,F401
from datasets import Base_Dataset, load_part_kp  # noqa: E402
from SLRT_metrics import translation_performance, islr_performance  # noqa: E402,F401

from release_data import read_split, load_captions, load_metadata, rtmpose_index, internal_id  # noqa: E402,F401


def absolute(path):
    return os.path.normpath(os.path.join(ORIG_CWD, path)) if path else path


def get_parser(description):
    """Upstream Uni-Sign arguments plus the release arguments."""
    p = argparse.ArgumentParser(description, parents=[utils.get_args_parser()])
    p.add_argument('--unisign_repo', default=os.environ.get('UNISIGN_REPO', ''), help='checkout of the official Uni-Sign repository')
    p.add_argument('--data_root', required=True, help='downloaded MoGloss release')
    p.add_argument('--cache_dir', default='', help='where the tar shard indexes are cached (default ~/.cache/mogloss_index)')
    return p


def translation_items(data_root, setting, split):
    """[(clip, caption, gloss)] of a translation split in list order; clips without a caption are dropped."""
    cap = load_captions(data_root, setting)
    return [(c, cap[c], '') for c in read_split(data_root, split) if cap.get(c)]


def to_device(src, dev):
    """Float tensors to float32 on the device (load_part_kp can return float64 for clips with few visible body points)."""
    for k in src:
        if isinstance(src[k], torch.Tensor):
            src[k] = (src[k].float() if src[k].is_floating_point() else src[k]).to(dev)
    return src


class LabelPoseDataset(Base_Dataset):
    """A clip list read as the upstream S2T_Dataset reads poses: at most max_length frames, drawn with random.sample and
    kept in temporal order, then load_part_kp. Used for fine-tuning, evaluation and WLASL."""

    def __init__(self, items, index, args):
        super().__init__()
        self.args = args
        self.rgb_support = False
        self.max_length = args.max_length
        self.index = index
        self.items = [it for it in items if it[0] in index]
        self.n_missing = len(items) - len(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip, text, gloss = self.items[i]
        z = self.index.load(clip)
        K, S = z['keypoints'], z['scores']
        dur = len(S)
        tmp = sorted(random.sample(range(dur), k=self.max_length)) if dur > self.max_length else list(range(dur))
        sk = [K[j][None] for j in tmp]
        cf = [S[j][None] for j in tmp]
        return clip, load_part_kp(sk, cf, force_ok=True), text, gloss, {}


class SegmentPoseDataset(Base_Dataset):
    """A clip list read as in our Uni-Sign pretraining runs: clips sorted by their internal id, at most max_length frames
    drawn with numpy and kept in temporal order, keypoints cast to float32 before load_part_kp. Used for the YouTube-ASL
    pretraining (setting A) and for training on all data (setting B)."""

    def __init__(self, items, index, args):
        super().__init__()
        self.args = args
        self.rgb_support = False
        self.max_length = args.max_length
        self.index = index
        self.items = sorted((it for it in items if it[0] in index), key=lambda it: internal_id(it[0]))
        self.n_missing = len(items) - len(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip, text, gloss = self.items[i]
        z = self.index.load(clip)
        K, S = z['keypoints'], z['scores']
        idx = np.arange(len(S))
        if len(idx) > self.max_length:
            idx = np.sort(np.random.choice(idx, self.max_length, replace=False))
        sk = [np.asarray(K[j], np.float32)[None] for j in idx]
        cf = [np.asarray(S[j], np.float32)[None] for j in idx]
        kp = {k: v.float() for k, v in load_part_kp(sk, cf, force_ok=True).items()}
        return clip, kp, text, gloss, {}
