# YouTube-ASL baseline reproduction

Our reproduction of the translation baseline of YouTube-ASL (Uthus et al., "YouTube-ASL: A Large-Scale, Open-Domain American Sign
Language-English Parallel Corpus", NeurIPS 2023 Datasets and Benchmarks): 85 keypoints per frame, a learned linear projection into
T5-v1.1-Base, pretraining on YouTube-ASL and fine-tuning on each target dataset (setting A of the paper).

The authors did not release training code; `train.py` follows the recipe of their paper. It needs PyTorch, transformers, sacrebleu and
the pretrained `google/t5-v1_1-base` (downloaded by transformers, or a local copy passed with `--t5`).

## Input

The 85 keypoints of the paper (42 hand points, 6 body points: shoulders, elbows, hips, and 37 face points: eyebrows, eyes, outer lips
and 3 outline points) are taken from our RTMPose keypoints (COCO-WholeBody indices are listed in `train.py`). Points with score below
0.3 are set to -10; the other points of a clip are min-max scaled per axis over the whole clip. At most 256 frames per clip.

`--data_root` is the MoGloss release (Hugging Face dataset `HenryH101/MoGloss`). `train.py` reads the setting-A keypoint folders of
`keypoints/rtmpose_2d` in place (an index of the tar members is built once and cached in `--cache_dir`, default
`~/.cache/mogloss_index`), the setting-A captions of `metadata/*.jsonl` (whitespace collapsed) and the lists of `splits/setting_a`.

## Commands

```bash
R=<MoGloss release>
```

Pretraining on the YouTube-ASL caption segments: Adafactor, constant learning rate 1e-3, batch 128. Validation on the 1,000 dev
segments every 10K steps (beam 5), early stop when the best dev BLEU gained less than 0.2 over the last 20K steps, at most 400K steps.
`best.pt` is the best dev checkpoint.
```bash
python train.py --data_root $R --train_split setting_a/pretrain_youtube_asl_train --val_split setting_a/pretrain_youtube_asl_dev \
  --steps 400000 --batch_size 128 --workers 10 --beams 5 --patience 20000 --min_gain 0.2 --val_every 10000 --run_dir runs/pretrain
```
The released checkpoint was trained in three consecutive pieces with the same data, batch and constant learning rate (to 94,100,
200,000 and 260,000 steps, each resumed from the `last.pt` of the previous piece). It stopped early at 260K steps; the best dev BLEU
was 18.09 at 250K steps, which is `best.pt`.

Fine-tuning for 20 epochs at batch 128 (How2Sign 4,851 steps, STEM 9,236, OpenASL 14,422), from `best.pt`. The last step decodes the
test list with beam 4.
```bash
for X in how2sign:4851 stem:9236 openasl:14422; do
  N=${X%%:*}; S=${X##*:}
  python train.py --data_root $R --init runs/pretrain/best.pt --train_split setting_a/finetune_${N}_train --test_split setting_a/test_$N \
    --steps $S --batch_size 128 --workers 10 --beams 4 --run_dir runs/ft_$N
done
python ../common/score_openasl_split.py runs/ft_openasl/test_predictions.tsv --data_root $R --out runs/ft_openasl/split_scores.json
```
`test_last.json` holds BLEU-1 to BLEU-4 of the last step (sacreBLEU, 13a tokenization, case-sensitive).

## Released weights and paper results

Weights are in the release under `weights/reproductions/youtube_asl_baseline/` (`{'model': state_dict}` of the `YTASLBaseline`
module in `train.py`; pass a file with `--init`).

| run | command output | release file | paper result (BLEU-4) |
|---|---|---|---|
| pretraining | `runs/pretrain/best.pt` | `youtube_asl_pretrain.pt` | dev 18.09 |
| How2Sign | `runs/ft_how2sign/last.pt` | `finetune_how2sign.pt` | 3.32 |
| ASL STEM Wiki | `runs/ft_stem/last.pt` | `finetune_stem.pt` | 3.19 |
| OpenASL | `runs/ft_openasl/last.pt` | `finetune_openasl.pt` | 13.23 (unseen 6.37, seen 16.10) |

## Deviations from the YouTube-ASL paper

From the reproduction table of our paper:
- 2D RTMPose keypoints instead of 3D MediaPipe points: 170 values per frame instead of 255.
- Beam 4 instead of 5 at test time (the shared protocol of setting A).

Further details of our runs:
- Pretraining uses the 602,243 YouTube-ASL caption segments of setting A (785 hours) and stops early on the dev BLEU (at 260K steps)
  instead of running a fixed 200K steps.
- Fine-tuning runs 20 epochs per dataset (the shared protocol of setting A) instead of 5,000 steps, and the last step is reported.
- Keypoints are on every second video frame, as in the paper.
- Pretraining validation uses beam 5, as in the paper. bf16 autocast.
