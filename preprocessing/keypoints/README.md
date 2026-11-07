# 2D keypoint extraction

`extract_rtmpose.py` produces the RTMPose keypoints used for every clip in the paper.

**Settings (keep them to match the released keypoints)**

| setting | value |
|---|---|
| library | [rtmlib](https://github.com/Tau-J/rtmlib) 0.0.13, `Wholebody(mode='lightweight', backend='onnxruntime', to_openpose=False)` |
| person detector | YOLOX-tiny, 416x416 |
| pose model | RTMW-dw-l-m, 256x192, 133 COCO-WholeBody keypoints |
| frames | every second frame (`--stride 2`) |
| person | one per frame, the detection with the highest mean keypoint score (zeros if nobody is detected) |
| coordinates | x / frame width, y / frame height (Uni-Sign pose format) |

These are the defaults of the script.

**Box-guided mode.** For multi-person OpenASL clips we skipped the detector and ran the pose model inside the official
OpenASL signer box (`--boxes`, a json that maps a clip name to a normalised `[x0, y0, x1, y1]` box, as in the OpenASL
bounding-box release). These files form the `openasl_setting_a_box` folder of the release.

## Usage

```bash
pip install rtmlib onnxruntime-gpu opencv-python numpy

# a folder of sentence clips -> one Uni-Sign pickle per clip
python extract_rtmpose.py clips/ --out_dir poses/ --format pkl

# a list of videos, split over 8 processes, release-style npz
python extract_rtmpose.py --list videos.txt --out_dir poses/ --format npz --shard 0 --nshards 8

# box-guided extraction for multi-person clips
python extract_rtmpose.py oa_clips/ --out_dir poses_box/ --boxes openasl_bbox.json
```

The run is resumable. Videos whose output already exists are skipped. The model files are downloaded by rtmlib on first use.

## Output

- `--format pkl`: `{'keypoints': [array (1, 133, 2) per kept frame], 'scores': [array (1, 133) per kept frame]}`, float32.
  This is the input format of the Uni-Sign code.
- `--format npz`: `keypoints` (T, 133, 2) and `scores` (T, 133), the layout of the released files.
  The released files keep the dtype of the original extraction, float32 for clips extracted one by one and float16 for
  YouTube-ASL and YouTube-SL-25, which were extracted per whole video (`--half --save_timing`) and then cut into caption
  segments.
- `--format both` writes both files.

**Cutting segments from a whole-video file.** With `--save_timing` the npz also holds `fps`, `stride` and `n_frames_src`.
A caption segment from `start` to `end` seconds is the slice we used in training:

```python
import math, numpy as np
z = np.load("video.npz"); T = len(z["scores"])
step = float(z["stride"]) / max(float(z["fps"]), 1e-6)          # seconds per kept frame
a = min(math.ceil(start / step), T - 1)
b = max(min(math.floor(end / step) + 1, T), a + 1)
kp, sc = z["keypoints"][a:b], z["scores"][a:b]
```
