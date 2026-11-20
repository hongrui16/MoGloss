# Reproductions

Code of the three methods we reproduce in the paper, trained on the same clips and tested on the same lists as our model.

| folder | method | runs in the paper |
|---|---|---|
| `uni-sign/` | Uni-Sign (Li et al., ICLR 2025), [official code](https://github.com/ZechengLi19/Uni-Sign) used unmodified + our data readers | setting A pose only and pose + RGB, setting B, WLASL |
| `youtube_asl_baseline/` | YouTube-ASL baseline (Uthus et al., NeurIPS 2023), T5-v1.1-Base on keypoints | setting A |
| `shubert/` | SHuBERT (Gueuwou et al.), released encoder + our ByT5-Base translation head | setting A, runs A, C1, C2 |
| `common/` | readers of the MoGloss release, video frame mapping, OpenASL unseen / seen scoring | |

**Uni-Sign is not re-implemented.** We run the official Uni-Sign code (https://github.com/ZechengLi19/Uni-Sign, commit `eed438b`) unmodified: its own model, pose features, batching, loss, argument defaults and metrics. Our scripts only match our data to it, reading the released keypoints, captions and splits into the inputs the official code expects. The one other difference is the launcher (one GPU with gradient accumulation instead of deepspeed, same clips per update).

Each folder has a README with the upstream dependency, the commands of every paper run stage by stage and the deviations from the
original paper.

All scripts read the MoGloss release (Hugging Face dataset `HenryH101/MoGloss`) given with `--data_root`: keypoints from the tar
shards in place, captions from `metadata/*.jsonl`, lists from `splits/`. Clip names are the released names. Runs that use RGB
(Uni-Sign pose + RGB, SHuBERT) also need the source videos, arranged as described in `common/videos.py`.

Translation is scored with sacreBLEU corpus BLEU-4 (13a tokenization, case-sensitive) on the last checkpoint with beam 4.
OpenASL is also scored on the test sentences from videos unseen and seen in YouTube-ASL:
```bash
python common/score_openasl_split.py <test_predictions.tsv> --data_root <release>
```
