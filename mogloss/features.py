"""Per-frame keypoint features.

A frame is the 69 RTMPose points used by Uni-Sign (9 body, 21 per hand, 18 face) with the Uni-Sign normalisation:
the body is scaled into [-1, 1] by its bounding box, hands are taken relative to the wrist and face points relative to
the nose, both divided by the body scale; points with confidence <= 0.3 are zeroed. Each point gives (x, y, confidence),
so a frame is 27 + 63 + 63 + 54 = 207 numbers.
"""
import copy
import numpy as np

FRAME_DIM = 207
PARTS = ('body', 'left', 'right', 'face')
SELECT = {
    'body': [0] + list(range(3, 11)),
    'left': list(range(91, 112)),
    'right': list(range(112, 133)),
    'face': list(range(23, 40))[::2] + list(range(83, 91)) + [53],
}


def crop_scale(motion, thr):
    """Uni-Sign crop_scale: square box around the confident points, mapped to [-1, 1]."""
    result = copy.deepcopy(motion)
    valid = motion[motion[..., 2] > thr][:, :2]
    if len(valid) < 4:
        return np.zeros(motion.shape), 0
    xmin, xmax, ymin, ymax = valid[:, 0].min(), valid[:, 0].max(), valid[:, 1].min(), valid[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape), 0
    xs, ys = (xmin + xmax - scale) / 2, (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result, scale


def frame_features(keypoints, scores, thr=0.3):
    """keypoints (T, 133, 2), scores (T, 133) -> (T, 207) float32."""
    K = np.asarray(keypoints, np.float32)
    S = np.asarray(scores, np.float32)
    T = len(K)
    out, scale = [], None
    for part in PARTS:
        kp = K[:, SELECT[part]].astype(np.float64)
        cf = S[:, SELECT[part]].astype(np.float64)
        if part in ('left', 'right'):
            kp = kp - kp[:, :1]
        elif part == 'face':
            kp = kp - kp[:, -1:]
        r = np.concatenate([kp, cf[..., None]], -1)
        if part == 'body':
            r, scale = crop_scale(r, thr)
        elif not scale:
            r = np.zeros(r.shape)
        else:
            r[..., :2] = r[..., :2] / scale
            r = np.clip(r, -1, 1)
            r[r[..., 2] <= thr] = 0
        out.append(r.astype(np.float32).reshape(T, -1))
    return np.concatenate(out, -1)


def hand_span(scores, thr=0.3, margin=2):
    """Frames in which either hand is seen (mean confidence of its 21 points > thr), widened by `margin` frames.
    Used to trim isolated-sign clips. Returns (start, end); the whole clip if no hand is seen."""
    S = np.asarray(scores, np.float32)
    hand = (S[:, 91:112].mean(1) > thr) | (S[:, 112:133].mean(1) > thr)
    det = np.where(hand)[0]
    if len(det) == 0:
        return 0, len(S)
    return max(0, det[0] - margin), min(len(S), det[-1] + 1 + margin)
