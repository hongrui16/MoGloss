# SHuBERT reproduction

Our reproduction of SHuBERT (Gueuwou et al., "SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream
Cluster Prediction") for ASL translation, as reported in the paper (setting A, runs A, C1 and C2).

SHuBERT releases its self-supervised encoder, the DINOv2 face and hand feature extractors and its preprocessing code, but not its
translation code. We use the released encoder and DINOv2 models, build the four input streams (face, left hand, right hand, upper
body) with the released crop code, and attach ByT5-Base through a learned softmax-weighted sum over the 12 encoder layers and a
linear layer, as described in the SHuBERT paper. The translation stage (`train.py`) is our implementation.

## Upstream dependency

- Code: https://github.com/ShesterG/SHuBERT (we used commit `cc1929326075bfbad7ad73159b2acf84356059bb`; the model classes in
  `fairseq/examples/shubert` are used unmodified; build fairseq as in the upstream README).
- Weights: `checkpoint_836_400000.pt` (SHuBERT), `dinov2face.pth`, `dinov2hand.pth`, from the weights folder linked in the upstream
  README (https://drive.google.com/drive/folders/1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV).
- DINOv2 model code is loaded with `torch.hub` (`facebookresearch/dinov2`, `dinov2_vits14_reg`), as in the upstream feature
  extraction. `--dinov2_hub` also accepts a local checkout.
- Stage 1 of runs C1 / C2 (optional, see below) needs the upstream preprocessing environment: mediapipe 0.10.14, ultralytics 8.2.74,
  decord 0.6.0, and the models `yolov8n.pt`, `face_landmarker.task`, `hand_landmarker.task` in one folder (`--mp_models`).
- Translation: PyTorch, transformers (tested with 4.57), sacrebleu, `google/byt5-base`. Set `USE_TF=0` if TensorFlow is installed.

## Data

- `--data_root`: the MoGloss release (Hugging Face dataset `HenryH101/MoGloss`). Used: `splits/setting_a/*`, `metadata/*.jsonl`
  (setting-A captions), `keypoints/rtmpose_2d` (run A, and the frame count of every clip), `keypoints/mediapipe` (runs C1 / C2).
- `--video_root`: the source videos, which are not redistributed. Layout and the keypoint-frame to video-frame rule are in
  `../common/videos.py`. All streams are built on the keypoint frames, i.e. every second video frame.

Stream files are written as `<streams>/<clip>.npz` with `face (T,384)`, `lhand (T,384)`, `rhand (T,384)`, `body (T,14)`, `T`.
The commands below use the default list of clips (all setting-A splits). Every builder takes `--shard k --nshards n` to split the work
by video, skips clips that are done, and can be restarted.

## Runs

| run | streams | weight decay 0.1 on | translation pretraining length |
|---|---|---|---|
| A  | `streams_rtmpose.py` (crops located by RTMPose keypoints) | all parameters | stopped by hand at about 133K updates, fine-tuned from best.pt (110K) |
| C1 | `mp_landmarks.py` + `streams_mediapipe.py` (SHuBERT preprocessing) | all parameters | 94,100 updates (20 epochs), fine-tuned from best.pt |
| C2 | as C1 | weight matrices only (`--hf_weight_decay`) | 94,100 updates (20 epochs), fine-tuned from best.pt |

Common variables:
```bash
R=<release>; V=<videos>; SH=<SHuBERT checkout>; W=<SHuBERT weights folder>
COMMON="--data_root $R --shubert_repo $SH --shubert_ckpt $W/checkpoint_836_400000.pt --byt5 google/byt5-base"
```

### Streams

Run A:
```bash
python streams_rtmpose.py --data_root $R --video_root $V --out_dir streams_a \
  --dinov2_face $W/dinov2face.pth --dinov2_hand $W/dinov2hand.pth
```

Runs C1 / C2. Stage 1 (signer box and MediaPipe landmarks) is already in the release (`keypoints/mediapipe`), so stage 2 reads it
directly:
```bash
python streams_mediapipe.py --data_root $R --video_root $V --out_dir streams_mp \
  --dinov2_face $W/dinov2face.pth --dinov2_hand $W/dinov2hand.pth
```
To recompute stage 1 (for example on new data), run `mp_landmarks.py` and pass its output to stage 2 with `--lm_dir`:
```bash
python mp_landmarks.py --data_root $R --video_root $V --out_dir landmarks --mp_models <models> --box yolo --workers 32
python streams_mediapipe.py ... --lm_dir landmarks
```
`--box yolo` is the person box of the SHuBERT preprocessing (YOLOv8n, `clips_bbox.py`). `--box rtmpose` takes the box from the
RTMPose keypoints instead (mean box of the confident points, widened 0.5 of its width left and right, 0.3 of its height on top and
0.1 below) and does not need YOLO. Each released landmark file records its box in `bbox`; files made with `--box rtmpose` also carry
`bbox_src = 'rtm'`.

### Translation pretraining on YouTube-ASL

128 utterances per update (`--micro_bs 8 --accum 16`), AdamW, peak learning rate 5e-4 (5e-5 for SHuBERT), 10K warmup, cosine
schedule over 250K updates, weight decay 0.1, label smoothing 0.2, validation on the 1,000 dev segments every 10K updates (beam 4).

```bash
PRE="--train_list setting_a/pretrain_youtube_asl_train --val_list setting_a/pretrain_youtube_asl_dev --steps 250000 \
  --micro_bs 8 --accum 16 --lr 5e-4 --warmup 10000 --beams 4 --val_every 10000 --save_every 1000 --workers 10"
# A: no --stop_at; the paper run was stopped by hand at about 133K updates (best.pt = 110K updates)
python train.py $COMMON $PRE --streams streams_a  --run_dir runs/a_pre
# C1
python train.py $COMMON $PRE --streams streams_mp --run_dir runs/c1_pre --stop_at 94100
# C2
python train.py $COMMON $PRE --streams streams_mp --run_dir runs/c2_pre --stop_at 94100 --hf_weight_decay
```
For run A the first 18K updates were made as 16 x 8 instead of 8 x 16 utterances (the same 128 per update).

### Fine-tuning per dataset

20 epochs, learning rate 1e-4 (1e-5 for SHuBERT), warmup 10% of the updates (How2Sign 485, STEM 924, OpenASL 1,442 of 4,840 /
9,220 / 14,420), cosine, last checkpoint decoded on the test list with beam 4.

```bash
for DS in how2sign:485 stem:924 openasl:1442; do
  N=${DS%%:*}; WU=${DS##*:}
  FT="--train_list setting_a/finetune_${N}_train --test_list setting_a/test_${N} --epochs 20 --micro_bs 8 --accum 16 --lr 1e-4 --warmup $WU --beams 4 --workers 10"
  python train.py $COMMON $FT --streams streams_a  --init runs/a_pre/best.pt  --run_dir runs/a_ft_$N
  python train.py $COMMON $FT --streams streams_mp --init runs/c1_pre/best.pt --run_dir runs/c1_ft_$N
  python train.py $COMMON $FT --streams streams_mp --init runs/c2_pre/best.pt --run_dir runs/c2_ft_$N --hf_weight_decay
done
```

`test_last.json` holds the BLEU-4 of the last checkpoint (sacreBLEU, 13a, case-sensitive). For OpenASL, the scores on the
sentences from videos unseen / seen in YouTube-ASL:
```bash
python ../common/score_openasl_split.py runs/c1_ft_openasl/test_predictions.tsv --data_root $R --out runs/c1_ft_openasl/split_scores.json
```

Paper results (BLEU-4, setting A): How2Sign / STEM / OpenASL all / unseen / seen = A 0.04 / 0.13 / 1.30 / 0.97 / 1.44,
C1 0.01 / 0.05 / 0.21 / 0.12 / 0.25, C2 0.35 / 0.39 / 0.46 / 0.25 / 0.52. The weights of these runs are not part of the release.

## Deviations from the SHuBERT paper

From the paper's reproduction table:
- Run A locates the face and hand crops and the body points with our RTMPose keypoints instead of YOLO + MediaPipe.
- Translation pretraining uses YouTube-ASL only (the 602,243 caption segments of setting A, 785 hours).
- Run C2 applies weight decay only to weight matrices, not to biases and normalization layers. The SHuBERT paper gives a weight decay
  of 0.1 without saying which parameters it applies to; runs A and C1 apply it to all parameters.

Details the SHuBERT paper does not specify, chosen by us:
- 128 utterances per update (the paper gives 2 per GPU with 8 accumulation steps and no GPU count).
- Gradient-norm clipping at 1.0, bf16 autocast, at most 1,024 frames per clip, at most 384 output bytes.
- Fine-tuning for 20 epochs with 10% warmup instead of 50K steps with 5K warmup (the shared protocol of setting A).
- Beam 4 instead of 5 (the shared protocol of setting A).
- Runs C1 and C2 stop translation pretraining after 20 epochs (94,100 updates) on the 250K-update schedule.
- Crops are cut from the RGB frame and given to DINOv2 directly, without the intermediate crop videos of the upstream scripts.
  Streams are built on every second video frame, as in the SHuBERT paper.

## Files

| file | role |
|---|---|
| `train.py` | translation pretraining and fine-tuning (SHuBERT + ByT5-Base) |
| `streams_rtmpose.py` | run A streams from RTMPose keypoints and video |
| `mp_landmarks.py` | runs C1 / C2, stage 1: signer box + MediaPipe landmarks (released as `keypoints/mediapipe`) |
| `streams_mediapipe.py` | runs C1 / C2, stage 2: crops from the landmarks + DINOv2 |
| `shubert_crops.py` | crop, body-feature and DINOv2 helpers (copies of the upstream preprocessing functions) |
