# MoGloss: Joint Motion and Pseudo-Gloss Pretraining for American Sign Language Understanding

Code for MoGloss, a motion encoder for American Sign Language (ASL) that reads 2D keypoints, one frame per token.
It is pretrained in two stages, masked motion modeling (MMM) followed by joint masked modeling of motion and
pseudo-glosses, and is then used for sign language translation (SLT) and isolated sign language recognition (ISLR).

Project page: **https://hongrui16.github.io/MoGloss** |
Paper: **[arXiv:2611.25333](https://arxiv.org/abs/2611.25333)** |
Data, splits and all model weights: **https://huggingface.co/datasets/HenryH101/MoGloss**

**If you use this code, the data or the weights, please cite our paper** (see [Citation](#citation)).

```
mogloss/           model, keypoint features, data access, translation utilities
scripts/           pretraining, translation, recognition, evaluation
preprocessing/     RTMPose keypoint extraction and pseudo-gloss generation for new data
reproductions/     our reproductions of Uni-Sign, the YouTube-ASL baseline and SHuBERT
```

## 1. Installation

```bash
conda create -n mogloss python=3.10 -y && conda activate mogloss
pip install torch --index-url https://download.pytorch.org/whl/cu128   # any recent CUDA build of PyTorch 2.x
pip install -r requirements.txt
```

All experiments in the paper ran on a single A100 80 GB GPU with bfloat16 autocast.

## 2. Data and weights on Hugging Face

### Download

```bash
pip install -U "huggingface_hub[cli]"
# everything (about 250 GB)
huggingface-cli download HenryH101/MoGloss --repo-type dataset --local-dir data/MoGloss
# or only what you need, for example setting B testing
huggingface-cli download HenryH101/MoGloss --repo-type dataset --local-dir data/MoGloss \
    --include "metadata/*" "splits/*" "keypoints/rtmpose_2d/asl_stem_wiki/*" "keypoints/rtmpose_2d/openasl/*" \
              "keypoints/rtmpose_2d/how2sign/*" "keypoints/rtmpose_2d/youtube_asl/*" \
              "weights/mogloss/setting_b/translation/mmm_joint/*"
```

All commands below take `--data_root data/MoGloss`. Relative list paths such as `splits/setting_b/test_stem.txt` are
resolved inside the data root.

### What is in it

| folder | content |
|---|---|
| `keypoints/rtmpose_2d/<dataset>/*.tar` | RTMPose whole-body keypoints, one `<clip>.npz` per clip: `keypoints` (T, 133, 2), `scores` (T, 133) |
| `keypoints/mediapipe/*.tar` | MediaPipe landmarks, input of the SHuBERT reproduction |
| `metadata/<dataset>.jsonl` | one line per clip: captions and pseudo-glosses for setting A and B, labels for isolated signs |
| `splits/` | every list of clips used in the paper |
| `weights/mogloss/` | our encoders, translation models and classifiers |
| `weights/reproductions/` | Uni-Sign and YouTube-ASL baseline weights |

Datasets: `youtube_asl`, `youtube_asl_raw_captions`, `youtube_sl25_ase`, `openasl`, `openasl_setting_a_box`, `how2sign`,
`asl_stem_wiki`, `asl_citizen`, `asl_signbank`, `wlasl2000`, `wlasl2000_annotation_cut`. Clip names are the original
names with `:` replaced by `__`. The Hugging Face page describes every field.

You do not need to unpack the tar files. `mogloss.data.ClipStore` reads clips directly from the shards. The first time a
dataset folder is opened, it builds an index of the shard and caches it. Set `MOGLOSS_CACHE=<dir>` to keep the index
outside the data folder.

```python
from mogloss.data import ClipStore, load_metadata, caption
store = ClipStore("data/MoGloss", setting="b")      # setting "a" reads the setting-A OpenASL extraction where it differs
kp, sc = store.raw("stem__10066524_0_0")              # (T, 133, 2), (T, 133)
x = store.features("stem__10066524_0_0")              # (T, 207) model input, one row per frame
meta = load_metadata("data/MoGloss")
print(caption(meta, "stem__10066524_0_0", "b"))
```

### The two settings

* **Setting A.** Translation pretraining on 785 hours of YouTube-ASL, then fine-tuning on How2Sign, ASL STEM Wiki and
  OpenASL separately (20 epochs, last checkpoint). Lists: `splits/setting_a/`.
* **Setting B.** One model on all 1,112 hours, tested without fine-tuning. Every clip linked to a test video is removed
  from training. Lists: `splits/setting_b/`.

| list | used by |
|---|---|
| `setting_a/pretrain_youtube_asl_train.txt`, `pretrain_youtube_asl_dev.txt` | stage 1 (MMM), setting A |
| `setting_b/train.txt`, `val.txt` | stage 1 (MMM), setting B |
| `setting_{a,b}/joint_train.txt`, `joint_val.txt` | stage 2 (joint pretraining) |
| `setting_{a,b}/translation_train.txt` | translation pretraining |
| `setting_a/pretrain_youtube_asl_dev_600.txt`, `setting_b/val_600.txt` | checkpoint selection during translation pretraining |
| `setting_b/translation_train_{50,25}pct.txt` | less translation data (setting B) |
| `setting_a/finetune_{how2sign,stem,openasl}_train.txt` | per-dataset fine-tuning (setting A) |
| `setting_a/test_*.txt`, `setting_b/test_*.txt` | test sets |
| `openasl_test_seen.txt`, `openasl_test_unseen.txt` | OpenASL test sentences whose video is / is not in YouTube-ASL |
| `islr/asl_citizen_*.txt`, `islr/wlasl{100,2000}_*.txt` | isolated sign recognition |

### Weights

Our checkpoints are PyTorch dictionaries with `model` and `config`. Encoders and classifiers also store the frame
statistics `mu` / `sd` used for standardisation.

| file | what it is |
|---|---|
| `setting_{a,b}/pretraining/encoder_mmm_only.pt` | stage 1 (MMM) |
| `setting_{a,b}/pretraining/mmm_joint_full.pt` | stage 2 model with the pseudo-gloss branch and vocabulary |
| `setting_{a,b}/pretraining/encoder_mmm_joint.pt` | encoder after both stages (the full method) |
| `setting_{a,b}/pretraining/encoder_joint_only.pt`, `joint_only_full.pt` | ablation: stage 2 from random weights |
| `setting_{a,b}/pretraining/encoder_no_pretraining.pt` | ablation: random weights |
| `setting_a/translation/<encoder>/youtube_asl_pretrain.pt` | translation model after YouTube-ASL |
| `setting_a/translation/<encoder>/finetune_{how2sign,stem,openasl}.pt` | per-dataset fine-tuning |
| `setting_b/translation/<encoder>/all_data.pt` | one model on all data |
| `setting_b/translation/data_size/{mmm_joint,mmm_only}_{50,25}pct.pt` | less translation data |
| `islr/asl_citizen/{no_pretraining,mmm_only,mmm_joint}.pt`, `islr/wlasl{100,2000}/mmm_joint.pt` | classifiers |

`<encoder>` is `mmm_joint`, `mmm_only`, `joint_only` or `no_pretraining`. The translation models load T5-base from
Hugging Face for its configuration and tokenizer.

## 3. Testing with the released weights

```bash
R=data/MoGloss; W=$R/weights/mogloss

# Setting B: one model, four test sets (Table 5)
python scripts/evaluate_translation.py --data_root $R --setting b --ckpt $W/setting_b/translation/mmm_joint/all_data.pt \
    --test_lists splits/setting_b/test_stem.txt splits/setting_b/test_openasl.txt \
                 splits/setting_b/test_how2sign.txt splits/setting_b/test_youtube_asl.txt --out_dir results/setting_b

# Setting A: fine-tuned models (Table 4); OpenASL also reports the seen / unseen split
for ds in how2sign stem openasl; do
  python scripts/evaluate_translation.py --data_root $R --setting a \
      --ckpt $W/setting_a/translation/mmm_joint/finetune_$ds.pt --test_lists splits/setting_a/test_$ds.txt
done

# Isolated sign recognition (Tables 7 and 8)
python scripts/train_islr.py --data_root $R --dataset asl_citizen --eval_ckpt $W/islr/asl_citizen/mmm_joint.pt
python scripts/train_islr.py --data_root $R --dataset wlasl100  --eval_ckpt $W/islr/wlasl100/mmm_joint.pt
python scripts/train_islr.py --data_root $R --dataset wlasl2000 --eval_ckpt $W/islr/wlasl2000/mmm_joint.pt
```

Replace `mmm_joint` by `mmm_only`, `joint_only` or `no_pretraining` for the ablation rows. Decoding uses beam 4 and
BLEU is sacreBLEU corpus BLEU-4. Small differences in the last digit can come from GPU type and library versions.

## 4. Training from scratch

The pipeline is the same for both settings; only the lists differ. Setting B:

```bash
R=data/MoGloss; LEX=preprocessing/pseudo_gloss/sign_lexicon.tsv

# Stage 1: masked motion modeling (100K steps)
python scripts/pretrain_mmm.py --data_root $R --setting b \
    --train_list splits/setting_b/train.txt --val_list splits/setting_b/val.txt --out_dir runs/b/mmm

# Stage 2: joint motion and pseudo-gloss modeling (80K steps), then export the encoder
python scripts/pretrain_joint.py --data_root $R --setting b --init runs/b/mmm/best.pt --lexicon $LEX --out_dir runs/b/joint
python scripts/export_encoder.py --joint runs/b/joint/best.pt --out runs/b/joint/encoder.pt

# Translation (196,100 steps, best checkpoint on val_600), tested on the four test sets
python scripts/train_translation.py --mode pretrain --data_root $R --setting b --encoder runs/b/joint/encoder.pt \
    --train_list splits/setting_b/translation_train.txt --val_list splits/setting_b/val_600.txt \
    --steps 196100 --out_dir runs/b/translation
python scripts/evaluate_translation.py --data_root $R --setting b --ckpt runs/b/translation/best.pt \
    --test_lists splits/setting_b/test_stem.txt splits/setting_b/test_openasl.txt \
                 splits/setting_b/test_how2sign.txt splits/setting_b/test_youtube_asl.txt
```

Setting A uses `--setting a`, the stage-1 lists `splits/setting_a/pretrain_youtube_asl_train.txt` and
`pretrain_youtube_asl_dev.txt`, translation pretraining on `splits/setting_a/translation_train.txt` with
`--val_list splits/setting_a/pretrain_youtube_asl_dev_600.txt --steps 250935`, and then fine-tuning per dataset:

```bash
for ds_steps in how2sign:12935 stem:24630 openasl:38458; do
  ds=${ds_steps%%:*}; st=${ds_steps##*:}
  python scripts/train_translation.py --mode finetune --data_root $R --setting a --init runs/a/translation/best.pt \
      --train_list splits/setting_a/finetune_${ds}_train.txt --test_list splits/setting_a/test_$ds.txt \
      --steps $st --out_dir runs/a/finetune_$ds
done
```

Fine-tuning runs 20 epochs and reports the last checkpoint (`test_scores.json`).

**Ablations.**
* MMM only: use `runs/b/mmm/best.pt` as `--encoder`.
* No pretraining: `python scripts/init_random_encoder.py --stats_from runs/b/mmm/best.pt --out runs/b/random/encoder.pt`, then use it as `--encoder`.
* Joint only: stage 2 from the random encoder for as many steps as both stages together, with the stage-1 learning rate:
  `python scripts/pretrain_joint.py ... --init runs/b/random/encoder.pt --steps 180000 --lr 1e-4`, then export.
* Less translation data: `--train_list splits/setting_b/translation_train_50pct.txt --steps 98050`
  (25 %: `translation_train_25pct.txt --steps 49025`).

**Isolated sign recognition** from a pretrained encoder (40 epochs, best validation epoch is tested):

```bash
python scripts/train_islr.py --data_root $R --dataset asl_citizen --encoder runs/b/joint/encoder.pt --out_dir runs/islr/asl_citizen
python scripts/train_islr.py --data_root $R --dataset wlasl2000   --encoder runs/b/joint/encoder.pt --out_dir runs/islr/wlasl2000
```

## 5. New data

`preprocessing/keypoints` extracts RTMPose keypoints in the same format, and `preprocessing/pseudo_gloss` generates
pseudo-glosses from captions with a local open-weight language model. See the README files in those folders.

## 6. Reproduced methods

`reproductions/` holds the code and commands of our reproductions of Uni-Sign, the YouTube-ASL baseline and SHuBERT
under the same data and protocol. Each folder has its own README.

**Uni-Sign is not re-implemented.** We run the official Uni-Sign code (https://github.com/ZechengLi19/Uni-Sign, commit `eed438b`) unmodified: its own model, pose features, batching, loss, argument defaults and metrics. Our scripts only match our data to it, reading the released keypoints, captions and splits into the inputs the official code expects. The one other difference is the launcher (one GPU with gradient accumulation instead of deepspeed, same clips per update).

## Citation

If you find this work useful, please cite

```bibtex
@article{hong2026mogloss,
  title   = {MoGloss: Joint Motion and Pseudo-Gloss Pretraining for American Sign Language Understanding},
  author  = {Hong, Rui and Ko{\v{s}}eck{\'a}, Jana},
  journal = {arXiv preprint arXiv:2611.25333},
  year    = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Data and weights on Hugging Face: CC BY-NC 4.0, and parts derived from a source dataset are
also subject to the license of that dataset.
