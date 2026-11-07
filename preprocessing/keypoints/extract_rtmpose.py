"""2D whole-body keypoints with rtmlib (RTMPose / RTMW, 133 COCO-WholeBody points), one output file per video.

Settings used for every clip of the paper: rtmlib mode 'lightweight' (YOLOX-tiny person detector + RTMW-dw-l-m 256x192),
onnxruntime backend, every second frame (--stride 2). One person per frame:
  default   the detector runs on every kept frame and the detection with the highest mean keypoint score is kept
            (all zeros when nobody is detected, so the frame count stays aligned with the video).
  --boxes   box-guided mode, the detector is skipped and the pose model runs inside a given signer box
            (used for multi-person OpenASL clips with the official OpenASL signer boxes). The json maps a clip name
            (video file name without extension) to a normalised box [x0, y0, x1, y1].

Coordinates are divided by the frame width and height (the Uni-Sign pose format). Output formats:
  pkl  {'keypoints': [(1, 133, 2) per kept frame], 'scores': [(1, 133) per kept frame]}, float32 (Uni-Sign loader format)
  npz  keypoints (T, 133, 2), scores (T, 133); float32, or float16 with --half; with --save_timing also fps, stride and
       n_frames_src, so that segments of a long video can be cut later by time (see README.md)

  python extract_rtmpose.py videos/ --out_dir poses/ --format pkl
  python extract_rtmpose.py --list videos.txt --out_dir poses/ --format npz --shard 0 --nshards 8
  python extract_rtmpose.py clips/ --out_dir poses_box/ --boxes openasl_bbox.json
"""
import argparse, json, os, pickle, sys, time
import numpy as np, cv2
try:
    import torch  # noqa: F401  (loads the CUDA / cuDNN libraries that onnxruntime-gpu needs)
except ImportError:
    pass
from rtmlib import Wholebody

VIDEO_EXT = ('.mp4', '.mkv', '.webm', '.mov', '.avi')


def one_person(kps, scs):
    """Keep the detection with the highest mean score; zeros when nothing is detected."""
    if kps is None or len(kps) == 0:
        return np.zeros((1, 133, 2), np.float32), np.zeros((1, 133), np.float32)
    if len(kps) > 1:
        i = int(np.argmax(scs.mean(-1)))
        kps, scs = kps[i:i + 1], scs[i:i + 1]
    return kps.astype(np.float32), scs.astype(np.float32)


def run_video(path, infer, stride):
    """Run infer(frame) on every stride-th frame. Returns (K, S, fps, n_frames_src) or None."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    K, S = [], []
    fi = -1
    while True:
        fi += 1
        if fi % stride:
            if not cap.grab(): break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        kps, scs = infer(np.uint8(frame), W, H)
        K.append(kps / np.array([W, H], np.float32)[None, None])   # (1, 133, 2), normalised
        S.append(scs)                                               # (1, 133)
    cap.release()
    return (K, S, fps, fi) if K else None


def save(out_base, fmts, K, S, fps, n_src, stride, half, save_timing):
    if 'pkl' in fmts:
        tmp = out_base + '.pkl.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({'keypoints': K, 'scores': S}, f)
        os.replace(tmp, out_base + '.pkl')
    if 'npz' in fmts:
        dt = np.float16 if half else np.float32
        arrs = {'keypoints': np.concatenate(K).astype(dt), 'scores': np.concatenate(S).astype(dt)}
        if save_timing:
            arrs.update(fps=np.float32(fps), stride=np.int32(stride), n_frames_src=np.int32(n_src))
        tmp = out_base + '.tmp.npz'
        np.savez(tmp, **arrs)
        os.replace(tmp, out_base + '.npz')


def collect_videos(inputs, list_file):
    vids = []
    for p in inputs:
        if os.path.isdir(p):
            vids += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith(VIDEO_EXT)]
        else:
            vids.append(p)
    if list_file:
        with open(list_file) as f:
            vids += [l.strip() for l in f if l.strip()]
    return vids


def main():
    ap = argparse.ArgumentParser(description='rtmlib whole-body 2D keypoints, one output per video')
    ap.add_argument('inputs', nargs='*', help='video files and / or folders of videos')
    ap.add_argument('--list', default=None, help='text file with one video path per line')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--format', default='pkl', choices=['pkl', 'npz', 'both'])
    ap.add_argument('--boxes', default=None, help='json {clip name: [x0, y0, x1, y1] normalised}; enables box-guided mode')
    ap.add_argument('--mode', default='lightweight', choices=['lightweight', 'balanced', 'performance'])
    ap.add_argument('--stride', type=int, default=2, help='keep every N-th frame')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--half', action='store_true', help='store npz arrays as float16')
    ap.add_argument('--save_timing', action='store_true', help='also store fps, stride and n_frames_src in the npz')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    fmts = ['pkl', 'npz'] if a.format == 'both' else [a.format]
    os.makedirs(a.out_dir, exist_ok=True)

    vids = collect_videos(a.inputs, a.list)
    vids = [v for i, v in enumerate(vids) if i % a.nshards == a.shard]
    if a.limit:
        vids = vids[:a.limit]
    if not vids:
        sys.exit('no input videos')

    wb = Wholebody(to_openpose=False, mode=a.mode, backend='onnxruntime', device=a.device)
    boxes = None
    if a.boxes:
        with open(a.boxes) as f:
            boxes = json.load(f)
        pose = wb.pose_model                  # detector skipped
    print(f'shard {a.shard}/{a.nshards}: {len(vids)} videos, mode={a.mode}, stride={a.stride}, '
          f'{"box-guided" if boxes is not None else "detector"}', flush=True)

    t0 = time.time(); frames = 0; done = skip = fail = 0
    for k, v in enumerate(vids):
        name = os.path.splitext(os.path.basename(v))[0]
        out_base = os.path.join(a.out_dir, name)
        if all(os.path.exists(f'{out_base}.{e}') and os.path.getsize(f'{out_base}.{e}') > 0 for e in fmts):
            skip += 1; continue
        if boxes is not None:
            if name not in boxes:
                fail += 1; print(f'  no box {name}', flush=True); continue
            x0, y0, x1, y1 = boxes[name]
            def infer(frame, W, H, b=(x0, y0, x1, y1)):
                kps, scs = pose(frame, bboxes=[[b[0] * W, b[1] * H, b[2] * W, b[3] * H]])
                return kps[:1].astype(np.float32), scs[:1].astype(np.float32)
        else:
            def infer(frame, W, H):
                return one_person(*wb(frame))
        r = run_video(v, infer, a.stride)
        if r is None:
            fail += 1; print(f'  FAIL {v}: cannot open or no frames', flush=True); continue
        K, S, fps, n_src = r
        save(out_base, fmts, K, S, fps, n_src, a.stride, a.half, a.save_timing)
        frames += len(K); done += 1
        if (k + 1) % 25 == 0 or k == len(vids) - 1:
            el = time.time() - t0
            print(f'[{k + 1}/{len(vids)}] done={done} skip={skip} fail={fail} frames={frames} '
                  f'{frames / max(el, 1e-6):.1f} fps {el / 60:.1f} min', flush=True)
    print(f'DONE shard {a.shard}: done={done} skip={skip} fail={fail} frames={frames} in {(time.time() - t0) / 60:.1f} min', flush=True)
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
