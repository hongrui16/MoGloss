"""Crop, body-feature and DINOv2 helpers shared by the SHuBERT stream builders.

Functions marked "SHuBERT" are copied from the released SHuBERT preprocessing (dataset/crop_hands.py, dataset/crop_face.py,
features/body_features.py, features/dinov2_features.py). Functions marked "RTMPose" replace MediaPipe / YOLO by our RTMPose
keypoints (run A). torch is imported lazily so that the CPU landmark workers do not load it.
"""
import numpy as np

THR = 0.3                      # RTMPose keypoint confidence threshold
FACE_IDX = [69, 168, 156, 118, 54, 299, 347, 336, 301, 164, 212, 432, 18]   # the face points cues_on_grey_background reads
MP2COCO = {0: 0, 2: 1, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10}             # MediaPipe pose index -> COCO-WholeBody index


# ---------- SHuBERT: dataset/crop_hands.py, dataset/crop_face.py ----------
def resize_frame(frame, frame_size):
    import cv2
    if frame is not None and frame.size > 0:
        return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
    else:
        return None


def crop_frame(image, bounding_box):
    x, y, w, h = bounding_box
    cropped_frame = image[y:y + h, x:x + w]
    return cropped_frame


def get_bounding_box(landmarks, image_shape, scale_factor=1.2):
    import cv2
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
    xb, yb, wb, hb = cv2.boundingRect(landmarks_px)
    box_size = max(wb, hb)
    half_size = box_size // 2
    x = center_x - half_size
    y = center_y - half_size
    w = box_size
    h = box_size
    w_padding = int((scale_factor - 1) * w / 2)
    h_padding = int((scale_factor - 1) * h / 2)
    x -= w_padding
    y -= h_padding
    w += 2 * w_padding
    h += 2 * h_padding
    return x, y, w, h


def adjust_bounding_box(bounding_box, image_shape):
    x, y, w, h = bounding_box
    ih, iw, _ = image_shape
    if x + w > iw:
        x = iw - w
    if y + h > ih:
        y = ih - h
    x = max(x, 0)
    y = max(y, 0)
    return x, y, w, h


def select_face(pose_landmarks, face_landmarks):
    nose_landmark_from_pose = pose_landmarks[0]
    nose_landmarks_from_face = []
    for i in range(0, len(face_landmarks)):
        nose_landmarks_from_face.append(face_landmarks[i][0])
    closest_nose_index = np.argmin([np.linalg.norm(np.array(nose_landmark_from_pose) - np.array(nose_landmark)) for nose_landmark in nose_landmarks_from_face])
    return face_landmarks[closest_nose_index]


def select_hands(pose_landmarks, hand_landmarks, image_shape):
    if hand_landmarks is None:
        return None, None
    left_wrist_from_pose = pose_landmarks[15]
    right_wrist_from_pose = pose_landmarks[16]
    wrist_from_hand = []
    for i in range(0, len(hand_landmarks)):
        wrist_from_hand.append(hand_landmarks[i][0])
    if right_wrist_from_pose is not None:
        right_hand_landmarks = hand_landmarks[0]
        minimum_distance = 100
        for i in range(0, len(hand_landmarks)):
            distance = np.linalg.norm(np.array(right_wrist_from_pose[0:2]) - np.array(wrist_from_hand[i][0:2]))
            if distance < minimum_distance:
                minimum_distance = distance
                right_hand_landmarks = hand_landmarks[i]
        if minimum_distance >= 0.1:
            right_hand_landmarks = None
    else:
        right_hand_landmarks = None
    if left_wrist_from_pose is not None:
        left_hand_landmarks = hand_landmarks[0]
        minimum_distance = 100
        for i in range(0, len(hand_landmarks)):
            distance = np.linalg.norm(np.array(left_wrist_from_pose[0:2]) - np.array(wrist_from_hand[i][0:2]))
            if distance < minimum_distance:
                minimum_distance = distance
                left_hand_landmarks = hand_landmarks[i]
        if minimum_distance >= 0.1:
            left_hand_landmarks = None
    else:
        left_hand_landmarks = None
    return left_hand_landmarks, right_hand_landmarks


def calculate_bounding_box(landmarks, indices, image_shape):
    x_coordinates = [landmarks[i][0] for i in indices]
    y_coordinates = [landmarks[i][1] for i in indices]
    left = min(x_coordinates)
    right = max(x_coordinates)
    top = min(y_coordinates)
    bottom = max(y_coordinates)
    return int(left * image_shape[1]), int(top * image_shape[0]), \
        int(right * image_shape[1]), int(bottom * image_shape[0])


def cues_on_grey_background(image, facial_landmarks):
    left_eye_indices = [69, 168, 156, 118, 54]
    right_eye_indices = [168, 299, 347, 336, 301]
    mouth_indices = [164, 212, 432, 18]
    image_shape = image.shape
    left_eye_box = calculate_bounding_box(facial_landmarks, left_eye_indices, image_shape)
    right_eye_box = calculate_bounding_box(facial_landmarks, right_eye_indices, image_shape)
    mouth_box = calculate_bounding_box(facial_landmarks, mouth_indices, image_shape)
    min_x = min(left_eye_box[0], right_eye_box[0], mouth_box[0])
    min_y = min(left_eye_box[1], right_eye_box[1], mouth_box[1])
    max_x = max(left_eye_box[2], right_eye_box[2], mouth_box[2])
    max_y = max(left_eye_box[3], right_eye_box[3], mouth_box[3])
    padding = 10
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(image.shape[1], max_x + padding)
    max_y = min(image.shape[0], max_y + padding)
    width = max_x - min_x
    height = max_y - min_y
    side_length = max(width, height)
    if width < side_length:
        extra = side_length - width
        min_x = max(0, min_x - extra // 2)
        max_x = min(image.shape[1], max_x + extra // 2)
    if height < side_length:
        extra = side_length - height
        min_y = max(0, min_y - extra // 2)
        max_y = min(image.shape[0], max_y + extra // 2)
    grey_background = np.ones((side_length, side_length, 3), dtype=np.uint8) * 128

    def crop_and_paste(src, dst, src_box, dst_origin):
        x1, y1, x2, y2 = src_box
        dx, dy = dst_origin
        crop = src[y1:y2, x1:x2]
        crop_height, crop_width = crop.shape[:2]
        dst[dy:dy + crop_height, dx:dx + crop_width] = crop
    crop_and_paste(image, grey_background, left_eye_box, (left_eye_box[0] - min_x, left_eye_box[1] - min_y))
    crop_and_paste(image, grey_background, right_eye_box, (right_eye_box[0] - min_x, right_eye_box[1] - min_y))
    crop_and_paste(image, grey_background, mouth_box, (mouth_box[0] - min_x, mouth_box[1] - min_y))
    return grey_background


# ---------- SHuBERT: features/body_features.py ----------
def normalize_pose_keypoints(pose_landmarks):
    left_shoulder = np.array(pose_landmarks[11][:2])
    right_shoulder = np.array(pose_landmarks[12][:2])
    left_eye = np.array(pose_landmarks[2][:2])
    nose = np.array(pose_landmarks[0][:2])
    head_unit = np.linalg.norm(right_shoulder - left_shoulder) / 2
    signing_space_width = 6 * head_unit
    signing_space_height = 7 * head_unit
    signing_space_top = left_eye[1] - 0.5 * head_unit
    signing_space_left = nose[0] - signing_space_width / 2
    translation_matrix = np.array([[1, 0, -signing_space_left], [0, 1, -signing_space_top], [0, 0, 1]])
    scale_matrix = np.array([[1 / signing_space_width, 0, 0], [0, 1 / signing_space_height, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
    transformation_matrix = shift_matrix @ scale_matrix @ translation_matrix
    normalized_keypoints = []
    for landmark in pose_landmarks:
        keypoint = np.array([landmark[0], landmark[1], 1])
        normalized_keypoint = transformation_matrix @ keypoint
        normalized_keypoints.append(normalized_keypoint[:2])
    return normalized_keypoints


# ---------- RTMPose replacements (run A) ----------
def signer_box(Kp, S, iw, ih, left_right=0.3, top=0.01, bottom=0.01):
    """Signer crop (x0, y0, w, h) in pixels from RTMPose keypoints: mean per-frame box of the points with score > 0.3
    (at least 5 points), widened, clamped to the frame. Run A uses 0.3 / 0.01 / 0.01 (the YOLO margins of SHuBERT)."""
    boxes = []
    for k, s in zip(Kp, S):
        ok = s > THR
        if ok.sum() >= 5:
            p = k[ok]
            boxes.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
    if not boxes:
        return 0, 0, iw, ih
    x1, y1, x2, y2 = np.mean(boxes, 0)
    w, h = x2 - x1, y2 - y1
    x1, x2, y1, y2 = max(0, x1 - left_right * w), min(iw, x2 + left_right * w), max(0, y1 - top * h), min(ih, y2 + bottom * h)
    return x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)


def body_feat(P):
    """normalize_pose_keypoints on the 7 joints; P = dict MediaPipe index -> (x, y) in crop-normalised coordinates."""
    ls, rs, le, nose = np.array(P[11]), np.array(P[12]), np.array(P[2]), np.array(P[0])
    hu = np.linalg.norm(rs - ls) / 2
    sw, sh = 6 * hu, 7 * hu
    top = le[1] - 0.5 * hu
    left = nose[0] - sw / 2
    if hu <= 1e-6:
        return None
    return np.array([[(P[j][0] - left) / sw - 0.5, (P[j][1] - top) / sh - 0.5] for j in (0, 11, 12, 13, 14, 15, 16)], np.float32).reshape(-1)


def face_composite(img, F):
    """F = (68, 2) iBUG face points in pixels -> 224x224 eyes + mouth composite on grey (cues_on_grey_background with boxes
    approximated from the 68 points)."""
    import cv2
    ih, iw = img.shape[:2]

    def eye(eye_idx, brow_idx, outer):
        e, b = F[eye_idx], F[brow_idx]
        ew = e[:, 0].max() - e[:, 0].min()
        eh = max(e[:, 1].max() - e[:, 1].min(), 1.0)
        gap = max(e[:, 1].mean() - b[:, 1].mean(), 1.0)
        xs = [F[outer, 0] - 0.5 * ew * (1 if outer == 36 else -1), F[27, 0]]
        return int(max(0, min(xs))), int(max(0, b[:, 1].min() - 0.5 * gap)), int(min(iw, max(xs))), int(min(ih, e[:, 1].max() + eh))
    bA = eye(list(range(36, 42)), list(range(17, 22)), 36)
    bB = eye(list(range(42, 48)), list(range(22, 27)), 45)
    mw = F[54, 0] - F[48, 0]
    bM = (int(max(0, F[48, 0] - 0.25 * mw)), int(max(0, F[33, 1])), int(min(iw, F[54, 0] + 0.25 * mw)), int(min(ih, (F[57, 1] + F[8, 1]) / 2)))
    mnx, mny, mxx, mxy = min(bA[0], bB[0], bM[0]), min(bA[1], bB[1], bM[1]), max(bA[2], bB[2], bM[2]), max(bA[3], bB[3], bM[3])
    pad = 10
    mnx, mny, mxx, mxy = max(0, mnx - pad), max(0, mny - pad), min(iw, mxx + pad), min(ih, mxy + pad)
    w, h = mxx - mnx, mxy - mny
    side = max(w, h)
    if side <= 1:
        return None
    if w < side:
        ex = side - w
        mnx = max(0, mnx - ex // 2)
        mxx = min(iw, mxx + ex // 2)
    if h < side:
        ex = side - h
        mny = max(0, mny - ex // 2)
        mxy = min(ih, mxy + ex // 2)
    g = np.full((side, side, 3), 128, np.uint8)
    for x1, y1, x2, y2 in (bA, bB, bM):
        if x2 <= x1 or y2 <= y1:
            continue
        c = img[y1:y2, x1:x2]
        dx, dy = x1 - mnx, y1 - mny
        dy0, dx0 = max(dy, 0), max(dx, 0)
        c = c[dy0 - dy:, dx0 - dx:]
        ch, cw = c.shape[:2]
        ch, cw = min(ch, side - dy0), min(cw, side - dx0)
        if ch > 0 and cw > 0:
            g[dy0:dy0 + ch, dx0:dx0 + cw] = c[:ch, :cw]
    return cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA)


def hand_crop(img, H):
    """get_bounding_box(scale 1.5) + adjust_bounding_box + crop + resize, on pixel points."""
    import cv2
    ih, iw = img.shape[:2]
    px = H.astype(int)
    cx, cy = px.mean(0).astype(int)
    xb, yb, wb, hb = cv2.boundingRect(px)
    s = max(wb, hb)
    x, y = cx - s // 2, cy - s // 2
    pad = int((1.5 - 1) * s / 2)
    x -= pad
    y -= pad
    s += 2 * pad
    if x + s > iw:
        x = iw - s
    if y + s > ih:
        y = ih - s
    x, y = max(x, 0), max(y, 0)
    c = img[y:y + s, x:x + s]
    return cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA) if c.size else None


def hand_ok(img, H, sc, shoulder_w):
    """Hand crop from the confident points only. A hand is missing when its mean score < 0.3, fewer than 5 points are
    confident, or its box side exceeds 0.75 shoulder widths (scattered points)."""
    ok = sc > THR
    if sc.mean() < THR or ok.sum() < 5:
        return None
    P = H[ok]
    side = max(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min())
    if shoulder_w > 0 and side > 0.75 * shoulder_w:
        return None
    return hand_crop(img, P)


# ---------- DINOv2 (SHuBERT: features/dinov2_features.py) ----------
def dino(path, dev, hub='facebookresearch/dinov2'):
    """Released dinov2face.pth / dinov2hand.pth (teacher weights) in a ViT-S/14 with registers."""
    import os
    import torch
    import torch.nn as nn
    src = 'local' if os.path.isdir(hub) else 'github'
    m = torch.hub.load(hub, 'dinov2_vits14_reg', pretrained=False, source=src)
    sd = {k.replace('backbone.', ''): v for k, v in torch.load(path, map_location='cpu', weights_only=False)['teacher'].items() if 'dino_head' not in k}
    m.pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    m.load_state_dict(sd, strict=True)
    return m.to(dev).eval()


def embed(model, crops, dev, bs=256):
    """(N, 384) float16 DINOv2 features of a list of 224x224 RGB uint8 crops (ImageNet normalisation, fp16 autocast)."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            x = torch.from_numpy(np.stack(crops[i:i + bs])).permute(0, 3, 1, 2).float().div(255)
            x = ((x - mean) / std).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                out.append(model(x).float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


# ---------- clip lists ----------
SETTING_A_SPLITS = ['setting_a/pretrain_youtube_asl_train', 'setting_a/pretrain_youtube_asl_dev',
                    'setting_a/finetune_how2sign_train', 'setting_a/finetune_stem_train', 'setting_a/finetune_openasl_train',
                    'setting_a/test_how2sign', 'setting_a/test_stem', 'setting_a/test_openasl']


def clip_list(data_root, video_root, splits, shard=0, nshards=1):
    """[(clip, video path)] of the given splits, one shard (by crc32 of the video path), ordered by video so that a decoder
    can be reused for consecutive clips of one video."""
    import sys
    import os
    import zlib
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
    from release_data import read_split
    from videos import clip_video
    clips = sorted({c for s in splits for c in read_split(data_root, s)})
    rows = [(c, clip_video(c, video_root)[0]) for c in clips]
    rows = [r for r in rows if zlib.crc32(r[1].encode()) % nshards == shard]
    return sorted(rows, key=lambda r: (r[1], r[0]))
