"""SHuBERT preprocessing, stage 1 (runs C1 / C2, CPU only): signer box + MediaPipe landmarks per clip.

The release ships the output of this stage (keypoints/mediapipe), so it is only needed for new data.
Per clip, on the keypoint frames of the release (every second video frame):
  1. signer box, --box yolo: YOLOv8n person boxes on every frame, widened 0.3 w left / right and 0.01 h top / bottom; one person
     -> mean box, several -> optical-flow tube selection (SHuBERT dataset/clips_bbox.py, crop_clip).
     --box rtmpose: mean box of the confident RTMPose keypoints, widened 0.5 w left / right, 0.3 h on top, 0.1 h below.
  2. landmarks on the cropped frame (SHuBERT dataset/kpe_mediapipe.py): MediaPipe Holistic (min_detection_confidence 0.1, its pose
     is used), Face Landmarker (num_faces 6), Hand Landmarker (num_hands 6, min_hand_detection_confidence 0.05). Hands are matched
     to the pose wrists and the face to the pose nose with SHuBERT's select_hands / select_face. A frame whose MediaPipe call fails
     reuses the previous frame's landmarks.
Output <out_dir>/<clip>.npz (the format of keypoints/mediapipe in the release):
  bbox (4,) int32 x0, y0, x1, y1 in the full frame (-1 = no crop); hw (2,) size of the frame MediaPipe saw;
  pose (T, 25, 2), lhand / rhand (T, 21, 2), face (T, 13, 2) in [0, 1] of the cropped frame, NaN = not detected;
  T; mp_* detection counts; bbox_src = 'rtm' for --box rtmpose; decoder = 'cv2' when decord failed and OpenCV was used.
Models: yolov8n.pt (Ultralytics), face_landmarker.task and hand_landmarker.task (MediaPipe), all in --mp_models.

  python mp_landmarks.py --data_root <release> --video_root <videos> --out_dir <landmarks> --mp_models <dir> --box yolo \
      [--shard k --nshards n --workers 32]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import shubert_crops as C  # noqa: E402
from videos import frame_indices  # noqa: E402

_M = {}


# ---------- SHuBERT: dataset/clips_bbox.py ----------
def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def find_target_bbox(bbox_arr, opts, iou_thr=0.5, len_ratio_thr=0.5):
    tubes = []
    num_rest = sum([len(x) for x in bbox_arr])
    while num_rest > 0:
        for i, bboxes in enumerate(bbox_arr):
            if len(bboxes) > 0:
                anchor = [i, bbox_arr[i].pop()]
                break
        tube = [anchor]
        for i in range(len(bbox_arr)):
            bboxes = bbox_arr[i]
            if anchor[0] == i or len(bboxes) == 0:
                continue
            ious = np.array([get_iou(anchor[1], bbox) for bbox in bboxes])
            j = ious.argmax()
            if ious[j] > iou_thr:
                target_bbox = bboxes.pop(j)
                tube.append([i, target_bbox])
        tubes.append(tube)
        num_rest = sum([len(x) for x in bbox_arr])
    max_val, max_tube = -1, None
    for itube, tube in enumerate(tubes):
        mean_val = 0
        for iframe, bbox in tube:
            x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            mean_val += opts[iframe][max(y0, 0): y1, max(x0, 0): x1].mean()
        mean_val /= len(tube)
        if len(tube) / len(opts) > len_ratio_thr:
            if mean_val > max_val:
                max_val, max_tube = mean_val, tube
    if max_tube is not None:
        target_bbox = np.array([bbox[1] for bbox in max_tube]).mean(axis=0).tolist()
    else:
        target_bbox = None
    return target_bbox, tubes


def yolo_signer_bbox(chunks, height, width, yolo):
    """crop_clip of clips_bbox.py on frames given as chunks: (x_0, y_0, x_1, y_1) or None. YOLO runs one frame at a time as in
    crop_clip; the optical flow (get_optical_flow) is computed from half-size grey frames only when several people are found."""
    import cv2
    up_exp, down_exp, left_exp, right_exp = 0.01, 0.01, 0.3, 0.3
    bboxes, grays = [], []
    for ch in chunks:
        for f in ch:
            r = yolo(f, verbose=False, device='cpu')[0]
            bboxes.append([box.xyxy[0].tolist() for box in r.boxes if box.cls == 0])
            g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            grays.append(cv2.resize(g, (int(f.shape[1] * 0.5), int(f.shape[0] * 0.5))))
    for i in range(len(bboxes)):
        for j in range(len(bboxes[i])):
            x0, y0, x1, y1 = bboxes[i][j]
            w, h = x1 - x0 + 1, y1 - y0 + 1
            bboxes[i][j] = [x0 - w * left_exp, y0 - h * up_exp, x1 + w * right_exp, y1 + h * down_exp]
    if max([len(x) for x in bboxes]) == 0:
        return None
    if max([len(x) for x in bboxes]) == 1:
        bboxes = list(filter(lambda x: len(x) == 1, bboxes))
        bbox = np.array(bboxes).mean(axis=0)[0].tolist()
        tubes = []
    else:
        opts = []
        prv = None
        for g in grays:
            if prv is not None:
                flow = cv2.calcOpticalFlowFarneback(prv, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mag = (255.0 * (mag - mag.min()) / max(float(mag.max() - mag.min()), 1)).astype(np.uint8)
                mag = cv2.resize(mag, (width, height))
            else:
                mag = np.zeros((height, width), dtype=np.uint8)
            prv = g
            opts.append(mag)
        bbox, tubes = find_target_bbox(bboxes, opts)
    if bbox is None and len(tubes) > 0:
        if max([len(x) for x in tubes]) > 0:
            total_sizes = []
            for tube in tubes:
                total_size = sum([(bbox[3] - bbox[0]) * (bbox[2] - bbox[1]) for _, bbox in tube])
                total_sizes.append(total_size)
            idx = np.array(total_sizes).argmax()
            bbox = np.array([x for _, x in tubes[idx]]).mean(axis=0).tolist()
    if bbox is None:
        return None
    return int(max(bbox[0], 0)), int(max(bbox[1], 0)), int(min(bbox[2], width)), int(min(bbox[3], height))


def rtmpose_signer_bbox(K, S, height, width):
    """(x_0, y_0, x_1, y_1) from the RTMPose keypoints, widened 0.5 w left / right, 0.3 h on top, 0.1 h below."""
    x0, y0, w, h = C.signer_box(K * np.array([width, height], np.float32), S, width, height, left_right=0.5, top=0.3, bottom=0.1)
    return int(max(x0, 0)), int(max(y0, 0)), int(min(x0 + w, width)), int(min(y0 + h, height))


# ---------- SHuBERT: dataset/kpe_mediapipe.py ----------
def detect_holistic(image, holistic):
    """landmarks of one (cropped) RGB frame, as lists of [x, y, z] in [0, 1] of that frame."""
    import mediapipe as mp
    results = holistic.process(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
    face_prediction = _M['face'].detect(mp_image)
    hand_prediction = _M['hand'].detect(mp_image)
    d = {}
    d['face_landmarks'] = [[[l.x, l.y, l.z] for l in f] for f in face_prediction.face_landmarks] if face_prediction.face_landmarks else None
    d['hand_landmarks'] = [[[l.x, l.y, l.z] for l in h] for h in hand_prediction.hand_landmarks] if hand_prediction.hand_landmarks else None
    d['pose_landmarks'] = [[[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]] if results.pose_landmarks else None
    return d


def landmarks_stream(chunks, bb, T):
    """per-frame landmark decisions of one clip (the frames are cut to the signer box first)."""
    import mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.1)
    pose = np.full((T, 25, 2), np.nan, np.float32)
    lh = np.full((T, 21, 2), np.nan, np.float32)
    rh = lh.copy()
    face = np.full((T, 13, 2), np.nan, np.float32)
    prev_res = None
    st = {'res_fail': 0, 'pose': 0, 'face': 0, 'lhand': 0, 'rhand': 0}
    t = 0
    hw = None
    for ch in chunks:
        for img in ch:
            if bb is not None:
                img = img[bb[1]:bb[3], bb[0]:bb[2]]
            img = np.ascontiguousarray(img)
            hw = img.shape[:2]
            try:
                res = detect_holistic(img, holistic)
                prev_res = res
            except Exception:
                st['res_fail'] += 1
                res = prev_res
            if res is None or res['pose_landmarks'] is None:
                t += 1
                continue
            st['pose'] += 1
            p = res['pose_landmarks'][0]
            pose[t] = np.array(p[0:25], np.float32)[:, :2]
            l, r = C.select_hands(p, res['hand_landmarks'], img.shape)
            if l is not None:
                lh[t] = np.array(l, np.float32)[:, :2]
                st['lhand'] += 1
            if r is not None:
                rh[t] = np.array(r, np.float32)[:, :2]
                st['rhand'] += 1
            if res['face_landmarks'] is not None:
                f = C.select_face(p, res['face_landmarks'])
                face[t] = np.array([f[i][:2] for i in C.FACE_IDX], np.float32)
                st['face'] += 1
            t += 1
    holistic.close()
    return pose, lh, rh, face, hw, st


# ---------- frame readers ----------
def read_cv2(video, fidx):
    """Frames by sequential OpenCV read (fallback when decord fails); a frame past the decodable end repeats the last one."""
    import cv2
    cap = cv2.VideoCapture(video)
    want = set(int(i) for i in fidx)
    got = {}
    i = 0
    last = int(max(fidx))
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    if not got:
        raise RuntimeError('cv2 read no frames')
    out, prev = [], None
    for j in fidx:
        prev = got.get(int(j), prev if prev is not None else got[min(got)])
        out.append(prev)
    return out


def init_worker(box, mp_models):
    import cv2
    cv2.setNumThreads(1)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    if box == 'yolo':
        from ultralytics import YOLO
        import torch
        _M['yolo'] = YOLO(os.path.join(mp_models, 'yolov8n.pt'))
        _M['yolo'](np.zeros((360, 640, 3), np.uint8), verbose=False, device='cpu')
        torch.set_num_threads(1)
    _M['face'] = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=os.path.join(mp_models, 'face_landmarker.task')), num_faces=6))
    _M['hand'] = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=os.path.join(mp_models, 'hand_landmarker.task')), num_hands=6,
        min_hand_detection_confidence=0.05))


_VR = [None, None]


def clip_landmarks(cid, video, K, S, fidx, box, use_cv2=False):
    if use_cv2:
        import cv2
        cap = cv2.VideoCapture(video)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        iw, ih = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        fidx = np.clip(fidx, 0, max(n - 1, 0))
        frames = read_cv2(video, fidx)
        chunks = lambda: (np.stack(frames[i:i + 64]) for i in range(0, len(frames), 64))  # noqa: E731
    else:
        from decord import VideoReader, cpu
        if _VR[0] != video:
            _VR[0], _VR[1] = video, VideoReader(video, ctx=cpu(0), num_threads=1)
        vr = _VR[1]
        n = len(vr)
        ih, iw = vr[0].shape[:2]
        fidx = np.clip(fidx, 0, n - 1)
        chunks = lambda: (vr.get_batch(list(fidx[i:i + 64])).asnumpy() for i in range(0, len(fidx), 64))  # noqa: E731
    if box == 'yolo':
        bb = yolo_signer_bbox(chunks(), ih, iw, _M['yolo'])
    else:
        bb = rtmpose_signer_bbox(K, S, ih, iw)
    if bb is not None and not (bb[2] > bb[0] and bb[3] > bb[1]):
        bb = None
    pose, lh, rh, face, hw, st = landmarks_stream(chunks(), bb, len(fidx))
    d = {'bbox': np.array(bb if bb is not None else (-1, -1, -1, -1), np.int32), 'hw': np.array(hw, np.int32), 'pose': pose, 'lhand': lh,
         'rhand': rh, 'face': face, 'T': np.int32(len(fidx))}
    if box == 'rtmpose':
        d['bbox_src'] = np.array('rtm')
    if use_cv2:
        d['decoder'] = np.array('cv2')
    d.update({f'mp_{k}': np.int32(v) for k, v in st.items()})
    return d


def do_clip(task):
    cid, video, K, S, video_root, box = task
    try:
        _, fidx = frame_indices(cid, video_root, len(S))
        try:
            d = clip_landmarks(cid, video, K, S, fidx, box)
        except Exception as e:
            if 'DECORDError' not in repr(e):
                raise
            _VR[0] = _VR[1] = None
            d = clip_landmarks(cid, video, K, S, fidx, box, use_cv2=True)
        return cid, d
    except Exception as e:
        return cid, f'{e!r}'[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='downloaded MoGloss release (keypoints give the frames and, for --box rtmpose, the box)')
    ap.add_argument('--video_root', required=True, help='source videos, layout in ../common/videos.py')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mp_models', required=True, help='folder with yolov8n.pt, face_landmarker.task, hand_landmarker.task')
    ap.add_argument('--box', choices=['yolo', 'rtmpose'], default='yolo')
    ap.add_argument('--splits', nargs='+', default=C.SETTING_A_SPLITS)
    ap.add_argument('--cache_dir', default=None, help='where the tar index is cached')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_tasks', type=int, default=25, help='clip groups per worker before it is replaced (bounds memory growth)')
    a = ap.parse_args()
    from release_data import rtmpose_index
    os.makedirs(a.out_dir, exist_ok=True)
    have = {e.name for e in os.scandir(a.out_dir)}
    rows = [r for r in C.clip_list(a.data_root, a.video_root, a.splits, a.shard, a.nshards) if r[0] + '.npz' not in have]
    print(f'{len(rows)} clips to do', flush=True)
    kp = rtmpose_index(a.data_root, 'setting_a', a.cache_dir)

    def tasks():
        for cid, video in rows:
            z = kp.load(cid)
            yield cid, video, np.asarray(z['keypoints'], np.float32), np.asarray(z['scores'], np.float32), a.video_root, a.box

    t0 = time.time()
    n = fr = fail = 0
    with Pool(a.workers, initializer=init_worker, initargs=(a.box, a.mp_models), maxtasksperchild=a.max_tasks) as pool:
        for cid, d in pool.imap_unordered(do_clip, tasks(), chunksize=4):
            if isinstance(d, str):
                fail += 1
                print('FAIL', cid, d, flush=True)
                continue
            out = os.path.join(a.out_dir, cid + '.npz')
            np.savez(out + '.tmp.npz', **d)
            os.replace(out + '.tmp.npz', out)
            n += 1
            fr += int(d['T'])
            if n % 500 == 0:
                print(f'[{n}/{len(rows)}] fail {fail} frames {fr} | {fr / max(time.time() - t0, 1e-6):.1f} frames/s', flush=True)
    print(f'DONE shard {a.shard}: {n} clips, {fail} failed, {fr} frames', flush=True)


if __name__ == '__main__':
    main()
