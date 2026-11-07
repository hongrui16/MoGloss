"""Video files of the released clips (needed only by the RGB-based reproductions).

No video is redistributed. Download the source datasets and arrange them under --video_root as
  youtube_asl/<video_id>.mp4                   whole YouTube-ASL videos (clips ytseg__<video_id>__<start>_<end>)
  how2sign/<name>.mp4                          How2Sign sentence clips, front view (clips h2s__<name>)
  asl_stem_wiki/<article>/<name>.mp4           ASL STEM Wiki clips (clips stem__<name>, article = name up to the first '_')
  openasl/<vid>.mp4                            OpenASL sentence clips cut with cut_openasl_clips.py
                                               (clips oa__<vid with ':' replaced by '__'>)

The keypoints were extracted on every second video frame (stride 2). Keypoint frame j of a clip is
  video frame 2 * j                            for How2Sign, STEM and OpenASL clips,
  video frame 2 * (ceil(start / step) + j)     for YouTube-ASL caption segments cut from the whole video,
                                               step = 2 / fps, fps = the frame rate reported by OpenCV (as float32).
"""
import math
import os
import re

import numpy as np

STRIDE = 2


def clip_video(clip, video_root):
    """(video path, segment start in seconds or None) of a released clip name."""
    if clip.startswith('ytseg__'):
        m = re.match(r'ytseg__(.+)__([\d.]+)_([\d.]+)$', clip)
        return os.path.join(video_root, 'youtube_asl', m.group(1) + '.mp4'), float(m.group(2))
    if clip.startswith('h2s__'):
        return os.path.join(video_root, 'how2sign', clip[len('h2s__'):] + '.mp4'), None
    if clip.startswith('stem__'):
        name = clip[len('stem__'):]
        return os.path.join(video_root, 'asl_stem_wiki', name.split('_')[0], name + '.mp4'), None
    if clip.startswith('oa__'):
        m = re.match(r'oa__(.+)-(\d+__\d+__[\d.]+)-(\d+__\d+__[\d.]+)$', clip)
        ts = [m.group(i).replace('__', ':') for i in (2, 3)]
        return os.path.join(video_root, 'openasl', f'{m.group(1)}-{ts[0]}-{ts[1]}.mp4'), None
    raise ValueError(f'no video rule for clip {clip}')


def video_fps(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = float(np.float32(cap.get(cv2.CAP_PROP_FPS) or 0.0))
    cap.release()
    return fps


def frame_indices(clip, video_root, n):
    """(video path, video frame index of each of the n keypoint frames)."""
    path, start = clip_video(clip, video_root)
    if start is None:
        return path, np.arange(n) * STRIDE
    step = STRIDE / max(video_fps(path), 1e-6)
    a = int(math.ceil(start / step))
    return path, (a + np.arange(n)) * STRIDE
