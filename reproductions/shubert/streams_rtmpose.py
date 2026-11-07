"""SHuBERT input streams located by RTMPose keypoints (run A).

Per clip, on the keypoint frames of the release (every second video frame):
  face   (T, 384)  DINOv2 (released dinov2face.pth) of the eyes + mouth composite on grey, boxes from the 68 face points
  lhand  (T, 384)  DINOv2 (released dinov2hand.pth) of the subject's left hand crop (COCO-WholeBody 91-111)
  rhand  (T, 384)  right hand (112-132)
  body   (T, 14)   SHuBERT normalize_pose_keypoints on nose, shoulders, elbows, wrists, normalised to a signer box from the keypoints
A face is missing when the mean face score < 0.3; a hand when its mean score < 0.3, fewer than 5 points are confident or its box is
wider than 0.75 shoulder widths. A missing crop repeats the previous one (black before the first), a missing body row repeats the
previous one (-9999 before the first), as in SHuBERT. Crops are cut from the RGB frame and fed to DINOv2 as RGB.
Output: <out_dir>/<clip>.npz with face, lhand, rhand, body, T, signer_box.

  python streams_rtmpose.py --data_root <release> --video_root <videos> --out_dir <streams_a> \
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
from release_data import rtmpose_index  # noqa: E402
from videos import frame_indices  # noqa: E402

_VR = None   # (video path, reader) of the previous clip


def streams(K, S, fidx, video, models, dev):
    from decord import VideoReader, cpu
    global _VR
    if _VR is None or _VR[0] != video:
        _VR = (video, VideoReader(video, ctx=cpu(0)))
    vr = _VR[1]
    n = len(vr)
    fidx = np.clip(fidx, 0, n - 1)
    ih, iw = vr[0].shape[:2]
    Kp = K * np.array([iw, ih], np.float32)
    x0, y0, cw, ch = C.signer_box(Kp, S, iw, ih)
    face, lh, rh, body = [], [], [], []
    pf = pl = pr = pb = None
    black = np.zeros((224, 224, 3), np.uint8)
    for t0 in range(0, len(fidx), 64):
        frames = vr.get_batch(list(fidx[t0:t0 + 64])).asnumpy()
        for t, img in enumerate(frames, t0):
            k, s = Kp[t], S[t]
            if s[23:91].mean() >= C.THR:
                c = C.face_composite(img, k[23:91])
                pf = c if c is not None else pf
            face.append(pf if pf is not None else black)
            sw = np.linalg.norm(k[5] - k[6]) if min(s[5], s[6]) > C.THR else 0.0
            c = C.hand_ok(img, k[91:112], s[91:112], sw)
            pl = c if c is not None else pl
            lh.append(pl if pl is not None else black)
            c = C.hand_ok(img, k[112:133], s[112:133], sw)
            pr = c if c is not None else pr
            rh.append(pr if pr is not None else black)
            if all(s[C.MP2COCO[j]] > C.THR for j in (0, 2, 11, 12)):
                P = {j: ((k[c][0] - x0) / cw, (k[c][1] - y0) / ch) for j, c in C.MP2COCO.items()}
                b = C.body_feat(P)
                pb = b if b is not None else pb
            body.append(pb if pb is not None else np.full(14, -9999, np.float32))
    return {'face': C.embed(models['face'], face, dev), 'lhand': C.embed(models['hand'], lh, dev), 'rhand': C.embed(models['hand'], rh, dev),
            'body': np.stack(body).astype(np.float32), 'T': np.int32(len(fidx)), 'signer_box': np.array([x0, y0, cw, ch], np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='downloaded MoGloss release')
    ap.add_argument('--video_root', required=True, help='source videos, layout in ../common/videos.py')
    ap.add_argument('--out_dir', required=True)
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
    rows = C.clip_list(a.data_root, a.video_root, a.splits, a.shard, a.nshards)
    kp = rtmpose_index(a.data_root, 'setting_a', a.cache_dir)
    models = {'face': C.dino(a.dinov2_face, dev, a.dinov2_hub), 'hand': C.dino(a.dinov2_hand, dev, a.dinov2_hub)}
    t0 = time.time()
    n = fr = fail = 0
    for i, (cid, video) in enumerate(rows):
        out = os.path.join(a.out_dir, cid + '.npz')
        if os.path.exists(out):
            continue
        try:
            z = kp.load(cid)
            K, S = np.asarray(z['keypoints'], np.float32), np.asarray(z['scores'], np.float32)
            _, fidx = frame_indices(cid, a.video_root, len(K))
            d = streams(K, S, fidx, video, models, dev)
            np.savez(out + '.tmp.npz', **d)
            os.replace(out + '.tmp.npz', out)
            n += 1
            fr += int(d['T'])
        except Exception as e:
            fail += 1
            print('FAIL', cid, repr(e)[:200], flush=True)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            el = time.time() - t0
            print(f'[{i + 1}/{len(rows)}] done {n} fail {fail} frames {fr} | {fr / max(el, 1e-6):.1f} frames/s | {el / 60:.1f} min', flush=True)
    print('DONE', a.shard, n, fail, fr, flush=True)


if __name__ == '__main__':
    main()
