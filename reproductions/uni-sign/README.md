# Uni-Sign reproduction

Our reproduction of Uni-Sign (Li et al., "Uni-Sign: Toward Unified Sign Language Understanding at Scale", ICLR 2025) under the
shared protocol of the paper.

**Uni-Sign is not re-implemented.** We run the official Uni-Sign code (https://github.com/ZechengLi19/Uni-Sign, commit `eed438b`) unmodified: its own model, pose features, batching, loss, argument defaults and metrics. We use only its code, not the keypoints or weights released by Uni-Sign. Our scripts only match our data to it, reading our own RTMPose keypoints, captions and splits into the inputs the official code expects. The one other difference is the launcher (one GPU with gradient accumulation instead of deepspeed, same clips per update).

Runs:

- **setting A, pose only**: translation pretraining on YouTube-ASL (stage 1), then fine-tuning on How2Sign, ASL STEM Wiki and
  OpenASL (stage 3);
- **setting A, pose + RGB**: the RGB stage of Uni-Sign (stage 2) on YouTube-ASL, started from the pose-only stage 1, then the same
  per-dataset fine-tuning;
- **setting B**: one pose-only model trained on all data and tested without fine-tuning;
- **WLASL**: the released WLASL recipe of Uni-Sign, started from the setting-B model.

All runs read the same RTMPose keypoints as our model (69 of the 133 points, selected and normalised by Uni-Sign's own
`load_part_kp`).

## Upstream dependency

The official code, https://github.com/ZechengLi19/Uni-Sign, used **unmodified** (we used commit
`eed438bcb49e30405cd6ccdfcccca330c134e830`). Our scripts import its model (`models.Uni_Sign`), pose features
(`datasets.load_part_kp`), RGB hand crops (`datasets.load_support_rgb_dict`), batching (`Base_Dataset.collate_fn`), argument
defaults (`utils.get_args_parser`) and metrics (`SLRT_metrics`). They replace its deepspeed launcher with a single-GPU loop
(gradient accumulation gives the same clips per update) and its data lists with readers of the MoGloss release.

Setup:
- clone the repository and put mT5-Base in `<repo>/pretrained_weight/mt5-base`, as its README says;
- install its requirements; deepspeed is not needed; decord is needed only for the pose + RGB runs
  (`train_rgb.py`; the EfficientNet-B0 ImageNet weights are downloaded by torchvision);
- pass the checkout with `--unisign_repo <repo>` (or set `UNISIGN_REPO`). Our scripts switch the working directory to the
  repository because its `config.py` uses relative paths; all our path arguments are resolved against the directory the script was
  started from.

All other arguments (`--batch-size`, `--epochs`, `--lr`, `--max_length`, `--finetune`, `--dataset`, `--task`, `--seed`, ...) are the
upstream ones with their upstream defaults (AdamW, weight decay 1e-4, cosine schedule, no warmup, label smoothing 0.2,
at most 256 frames per clip, seed 42).

## Data

`--data_root` is the MoGloss release (Hugging Face dataset `HenryH101/MoGloss`). The scripts read
`keypoints/rtmpose_2d/<folder>/*.tar` in place (an index of the tar members is built once and cached in `--cache_dir`, default
`~/.cache/mogloss_index`), the captions in `metadata/*.jsonl` and the lists in `splits/`. Setting A reads the setting-A captions and
the box-guided keypoints of the multi-person OpenASL clips (`openasl_setting_a_box`), setting B the setting-B captions and the plain
`openasl` folder (see `../common/release_data.py`).

The pose + RGB runs also need the source videos (`--video_root`, layout in `../common/videos.py`). The keypoints are on every second
video frame; a pose frame chosen by Uni-Sign is mapped to its video frame for the hand crops.

## Commands

```bash
US=<Uni-Sign checkout>; R=<MoGloss release>; V=<videos>
COM="--unisign_repo $US --data_root $R --dataset OpenASL --task SLT --lr 3e-4"
```

### Setting A, pose only

Stage 1, YouTube-ASL caption segments, 20 epochs, 512 clips per update. Resumable with `--resume`.
```bash
python pretrain.py $COM --setting setting_a --train_split setting_a/pretrain_youtube_asl_train \
  --dev_split setting_a/pretrain_youtube_asl_dev --batch-size 16 --accum 32 --epochs 20 --num_workers 8 \
  --out_dir_ft runs/stage1 --resume
```
Stage 3, per dataset (`how2sign`, `stem`, `openasl`), 20 epochs, 32 clips per update, last epoch kept:
```bash
N=how2sign
python finetune.py $COM --finetune runs/stage1/final_checkpoint.pth --train_split setting_a/finetune_${N}_train \
  --batch-size 8 --accum 4 --epochs 20 --num_workers 6 --out_dir_ft runs/ft_$N
python evaluate.py $COM --finetune runs/ft_$N/final_checkpoint.pth --split setting_a/test_$N \
  --batch-size 8 --num_workers 4 --out_dir_eval runs/ft_$N/test
```
OpenASL, all / unseen / seen in YouTube-ASL:
```bash
python ../common/score_openasl_split.py runs/ft_openasl/test/test_predictions.tsv --data_root $R
```

### Setting A, pose + RGB

Stage 2 from the pose-only stage 1, YouTube-ASL, 5 epochs, 128 clips per update, dev BLEU after every epoch:
```bash
python train_rgb.py $COM --stage 2 --init runs/stage1/final_checkpoint.pth --video_root $V \
  --train_split setting_a/pretrain_youtube_asl_train --dev_split setting_a/pretrain_youtube_asl_dev \
  --epochs 5 --batch-size 16 --accum 8 --num_workers 14 --out_dir_ft runs/rgb_stage2
```
The released stage-2 checkpoint made its first 2,500 of 23,525 updates with `--batch-size 4 --accum 32` and the rest with
`--batch-size 16 --accum 8` (128 clips per update in both). To repeat this, run the first command with `--batch-size 4 --accum 32
--save_every 500` until update 2,500, then start the second one with `--resume_from <first run>/last.pth`.

Stage 3, per dataset, 20 epochs, 32 clips per update; the last epoch decodes the test split:
```bash
N=how2sign
python train_rgb.py $COM --stage 3 --init runs/rgb_stage2/final_checkpoint.pth --video_root $V \
  --train_split setting_a/finetune_${N}_train --test_split setting_a/test_$N \
  --epochs 20 --batch-size 8 --accum 4 --num_workers 14 --out_dir_ft runs/rgb_ft_$N
python ../common/score_openasl_split.py runs/rgb_ft_openasl/test_predictions.tsv --data_root $R   # OpenASL only
```

### Setting B, all data

20 epochs, 512 clips per update, from mT5-Base and random pose encoders. The dev set is `setting_b/val` without its STEM clips
(our run had no keypoint files for them). After training the test splits are decoded with the last epoch and with the best dev
epoch; the paper reports the last epoch.
```bash
python pretrain.py $COM --setting setting_b --train_split setting_b/train --dev_split setting_b/val --dev_exclude_prefix stem__ \
  --test_splits setting_b/test_stem,setting_b/test_openasl,setting_b/test_how2sign,setting_b/test_youtube_asl \
  --batch-size 16 --accum 32 --epochs 20 --num_workers 8 --out_dir_ft runs/setting_b --resume
```
The paper's OpenASL number of setting B was scored with the evaluation script:
```bash
python evaluate.py $COM --setting setting_b --finetune runs/setting_b/final_checkpoint.pth --split setting_b/test_openasl \
  --batch-size 8 --num_workers 4 --out_dir_eval runs/setting_b/test_openasl
```

### WLASL

The released WLASL recipe of Uni-Sign (gloss generated as text, exact match), from the setting-B model, 20 epochs, 32 clips per
update, at most 64 frames, best dev epoch scored on test. Input: `keypoints/rtmpose_2d/wlasl2000_annotation_cut` (the WLASL files cut
to the annotated span), lists `splits/islr/wlasl{100,2000}_*`, labels from `metadata/wlasl2000.jsonl`.
```bash
for S in wlasl100 wlasl2000; do
  python finetune_islr.py --unisign_repo $US --data_root $R --dataset WLASL --subset $S --finetune runs/setting_b/final_checkpoint.pth \
    --batch-size 8 --accum 4 --epochs 20 --lr 3e-4 --max_length 64 --num_workers 6 --out_dir_ft runs/$S
done
```

## Released weights and paper results

Weights are in the release under `weights/reproductions/uni-sign/` (`{'model': state_dict}`, load with the upstream `Uni_Sign`).

| run | command output | release file | paper result |
|---|---|---|---|
| setting A, stage 1 | `runs/stage1/final_checkpoint.pth` | `setting_a_pose/youtube_asl_pretrain.pth` | dev BLEU-4 14.41 (best 14.47, epoch 18) |
| setting A, pose, How2Sign | `runs/ft_how2sign/final_checkpoint.pth` | `setting_a_pose/finetune_how2sign.pth` | 9.28 |
| setting A, pose, STEM | `runs/ft_stem/final_checkpoint.pth` | `setting_a_pose/finetune_stem.pth` | 19.85 |
| setting A, pose, OpenASL | `runs/ft_openasl/final_checkpoint.pth` | `setting_a_pose/finetune_openasl.pth` | 21.56 (unseen 15.42, seen 24.19) |
| setting A, RGB stage 2 | `runs/rgb_stage2/final_checkpoint.pth` | `setting_a_pose_rgb/youtube_asl_rgb_stage2.pth` | dev BLEU-4 16.24 (best 16.43, epoch 4) |
| setting A, pose + RGB, How2Sign | `runs/rgb_ft_how2sign/final_checkpoint.pth` | `setting_a_pose_rgb/finetune_how2sign.pth` | 9.55 |
| setting A, pose + RGB, STEM | `runs/rgb_ft_stem/final_checkpoint.pth` | `setting_a_pose_rgb/finetune_stem.pth` | 20.51 |
| setting A, pose + RGB, OpenASL | `runs/rgb_ft_openasl/final_checkpoint.pth` | `setting_a_pose_rgb/finetune_openasl.pth` | 22.36 (unseen 16.48, seen 24.71) |
| setting B | `runs/setting_b/final_checkpoint.pth` | `setting_b/all_data.pth` | STEM 3.43, OpenASL 9.57, How2Sign 3.35, YouTube-ASL 6.33 |
| WLASL100 | `runs/wlasl100/best_checkpoint.pth` | `islr/wlasl100.pth` | 84.5 (per-instance top-1) |
| WLASL2000 | `runs/wlasl2000/best_checkpoint.pth` | `islr/wlasl2000.pth` | 47.3 |

Translation results are BLEU-4 (sacreBLEU, 13a tokenization, case-sensitive) of the last epoch with beam 4.

## Deviations from the Uni-Sign paper

From the reproduction table of our paper:
- The pretraining script of Uni-Sign is not released. Stage 1 uses the translation loss of the fine-tuning code on the pretraining
  data, with the stage-1 recipe of the paper (20 epochs, 512 clips per update, lr 3e-4).
- Pretraining uses YouTube-ASL only (the 602,243 caption segments of setting A, 785 hours).
- Keypoints come from the lightweight RTMPose model of rtmlib on every second frame (the same keypoints as our model and all
  keypoint-based reproductions).

Further details of our runs:
- One GPU with gradient accumulation instead of several GPUs with deepspeed; the clips per update follow the paper (512 in stage 1
  and setting B, 128 in stage 2, 32 in fine-tuning and on WLASL).
- Fine-tuning in setting A has no dev set and keeps the last epoch (no selection on the test set).
- Pose tensors are cast to float32 before the model; `load_part_kp` returns float64 for clips with fewer than 4 visible body points,
  which fails under bf16 autocast.
- WLASL clips are cut to their annotated span; one WLASL2000 training clip whose span lies outside its video has no keypoints and
  is skipped.

## Files

| file | role |
|---|---|
| `unisign_setup.py` | loads the upstream code; release readers (`LabelPoseDataset`, `SegmentPoseDataset`) |
| `pretrain.py` | stage 1 of setting A and the setting-B run (translation from mT5-Base and random pose encoders) |
| `finetune.py` | stage 3 of setting A, pose only |
| `evaluate.py` | test decoding of a pose-only checkpoint |
| `train_rgb.py` | pose + RGB, stage 2 and stage 3 |
| `finetune_islr.py` | WLASL isolated sign recognition |
