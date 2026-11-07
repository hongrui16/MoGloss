"""SHuBERT preprocessing, stage 2 (runs C1 / C2, GPU): the four input streams from the MediaPipe landmarks.

Per clip: decode the keypoint frames of the release (every second video frame), cut them to the stored signer box, cut the hand
and face crops from the stored landmarks with SHuBERT's crop_hands / crop_face rules (hand box scale 1.5, eyes + mouth on grey,
224x224), carry the previous crop / body row when a frame has none (black / -9999 before the first), body = SHuBERT
normalize_pose_keypoints on joints 0, 11-16, then the released DINOv2 face / hand models.
Landmarks are read from the release (keypoints/mediapipe) or, with --lm_dir, from the output folder of mp_landmarks.py.
Output: <out_dir>/<clip>.npz with face, lhand, rhand, body, T, signer_box and the mp_* counts.

  python streams_mediapipe.py --data_root <release> --video_root <videos> --out_dir <streams_mp> \
      --dinov2_face dinov2face.pth --dinov2_hand dinov2hand.pth [--shard k --nshards n]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import shubert_crops as C  # noqa: E402
from release_data import ShardIndex  # noqa: E402
from videos import frame_indices  # noqa: E402


def crops_from_lm(frames, lm, state):
    """per-frame crops of one chunk; state carries the previous crops / body row across chunks."""
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    bb = lm['bbox']
    out = {'face': [], 'lhand': [], 'rhand': [], 'body': []}
    for img in frames:
        t = state['t']
        state['t'] += 1
        if bb[0] >= 0:
            img = img[bb[1]:bb[3], bb[0]:bb[2]]
        img = np.ascontiguousarray(img)
        if np.isnan(lm['pose'][t, 0, 0]):
            for k in ('face', 'lhand', 'rhand'):
                out[k].append(state[k] if state[k] is not None else black)
            out['body'].append(state['body'] if state['body'] is not None else np.full((7, 2), -9999).flatten())
            continue
        for k in ('lhand', 'rhand'):
            if not np.isnan(lm[k][t, 0, 0]):
                c = C.resize_frame(C.crop_frame(img, C.adjust_bounding_box(C.get_bounding_box(lm[k][t].tolist(), img.shape, scale_factor=1.5), img.shape)), (224, 224))
                if c is not None:
                    state[k] = c
            out[k].append(state[k] if state[k] is not None else black)
        if not np.isnan(lm['face'][t, 0, 0]):
            F = np.zeros((478, 2), np.float32)
            F[C.FACE_IDX] = lm['face'][t]
            c = C.resize_frame(C.cues_on_grey_background(img, F.tolist()), (224, 224))
            if c is not None:
                state['face'] = c
        out['face'].append(state['face'] if state['face'] is not None else black)
        b = C.normalize_pose_keypoints(lm['pose'][t].tolist())
        state['body'] = np.array([b[j] for j in (0, 11, 12, 13, 14, 15, 16)]).flatten()
        out['body'].append(state['body'])
    return out


_VR = None   # (video path, reader) of the previous clip


def streams(lm, fidx, video, models, dev):
    global _VR
    from decord import VideoReader, cpu
    if _VR is None or _VR[0] != video:
        _VR = (video, VideoReader(video, ctx=cpu(0)))
    vr = _VR[1]
    fidx = np.clip(fidx, 0, len(vr) - 1)
    assert len(fidx) == int(lm['T']), (len(fidx), int(lm['T']))
    state = {'t': 0, 'face': None, 'lhand': None, 'rhand': None, 'body': None}
    emb = {'face': [], 'lhand': [], 'rhand': []}
    body = []
    for i in range(0, len(fidx), 64):
        c = crops_from_lm(vr.get_batch(list(fidx[i:i + 64])).asnumpy(), lm, state)
        emb['face'].append(C.embed(models['face'], c['face'], dev))
        emb['lhand'].append(C.embed(models['hand'], c['lhand'], dev))
        emb['rhand'].append(C.embed(models['hand'], c['rhand'], dev))
        body += c['body']
    d = {k: np.concatenate(v) for k, v in emb.items()}
    d.update({'body': np.stack(body).astype(np.float32), 'T': np.int32(len(fidx)), 'signer_box': np.array(lm['bbox'], np.float32)})
    d.update({k: lm[k] for k in lm if k.startswith('mp_')})
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='downloaded MoGloss release')
    ap.add_argument('--video_root', required=True, help='source videos, layout in ../common/videos.py')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--lm_dir', default='', help='landmarks written by mp_landmarks.py (default: keypoints/mediapipe of the release)')
    ap.add_argument('--dinov2_face', required=True, help='dinov2face.pth of the SHuBERT release')
    ap.add_argument('--dinov2_hand', required=True, help='dinov2hand.pth of the SHuBERT release')
    ap.add_argument('--dinov2_hub', default='facebookresearch/dinov2', help='torch.hub repo or a local checkout of it')
    ap.add_argument('--splits', nargs='+', default=C.SETTING_A_SPLITS)
    ap.add_argument('--cache_dir', default=None, help='where the tar index is cached')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    dev = 'cuda'
    os.makedirs(a.out_dir, exist_ok=True)
    have = {e.name for e in os.scandir(a.out_dir)}
    rows = [r for r in C.clip_list(a.data_root, a.video_root, a.splits, a.shard, a.nshards) if r[0] + '.npz' not in have]
    if a.lm_dir:
        def load_lm(cid):
            z = np.load(os.path.join(a.lm_dir, cid + '.npz'))
            return {k: z[k] for k in z.files}
    else:
        idx = ShardIndex(a.data_root, [os.path.join('keypoints', 'mediapipe')], a.cache_dir)
        load_lm = idx.load
    print(f'{len(rows)} clips to do', flush=True)
    models = {'face': C.dino(a.dinov2_face, dev, a.dinov2_hub), 'hand': C.dino(a.dinov2_hand, dev, a.dinov2_hub)}
    t0 = time.time()
    n = fr = fail = 0
    for i, (cid, video) in enumerate(rows):
        out = os.path.join(a.out_dir, cid + '.npz')
        try:
            lm = load_lm(cid)
            _, fidx = frame_indices(cid, a.video_root, int(lm['T']))
            d = streams(lm, fidx, video, models, dev)
            np.savez(out + '.tmp.npz', **d)
            os.replace(out + '.tmp.npz', out)
            n += 1
            fr += int(d['T'])
        except Exception as e:
            fail += 1
            print('FAIL', cid, repr(e)[:200], flush=True)
        if (i + 1) % 500 == 0 or i + 1 == len(rows):
            el = time.time() - t0
            print(f'[{i + 1}/{len(rows)}] done {n} fail {fail} frames {fr} | {fr / max(el, 1e-6):.1f} frames/s | {el / 60:.1f} min', flush=True)
    print('DONE', a.shard, n, fail, fr, flush=True)


if __name__ == '__main__':
    main()
