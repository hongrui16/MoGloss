# Preprocessing

| folder | what it does |
|---|---|
| [`keypoints/`](keypoints/README.md) | 2D whole-body keypoints (rtmlib RTMPose, lightweight, every second frame) for every clip |
| [`pseudo_gloss/`](pseudo_gloss/README.md) | pseudo-glosses from the English captions with a local LLM (Qwen3.5-27B) and a sign lexicon |

The outputs of both steps are released with the dataset (keypoints, captions and pseudo-glosses), so running them is only
needed for new videos or captions.
