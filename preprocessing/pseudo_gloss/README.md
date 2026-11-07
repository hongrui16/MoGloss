# Pseudo-gloss generation

Every English caption is rewritten by a local open-weight LLM into a sequence of sign-like units, one per sign, in caption
order. Each unit has a tag.

| tag | meaning |
|---|---|
| `L` | an entry of the sign lexicon, written exactly as in the lexicon (multi-word entries such as `all day` are one unit) |
| `W` | another content word, as its base form |
| `P` | a pronoun or pointing (`he`, `this`, `there`, ...) |
| `FS` | a name or acronym that would be fingerspelled |

Articles, the copula, auxiliaries, the infinitive *to* and fillers are dropped. Negation and question words are kept.

## Files

| file | content |
|---|---|
| `prompt.txt` | the system prompt (rules and two worked examples). The lexicon keys are appended to it, joined by `", "`. |
| `sign_lexicon.tsv` | the 3,339 lexicon keys (ASL SignBank glosses and the 2,731 ASL Citizen signs, lowercased, spelling variants merged), in the order used in the prompt |
| `generate_pseudo_gloss.py` | captions -> jsonl with the tagged units (vLLM) |
| `parse_units.py` | parsing of the LLM reply, and pseudo-gloss jsonl -> the unit lists used in training |

## Generation

We used Qwen3.5-27B with vLLM 0.30, bfloat16, greedy decoding (temperature 0, at most 256 new tokens), thinking mode off,
one request per unique caption. A single 80 GB GPU is enough (`--tp 2` for two 40 GB GPUs).

```bash
pip install vllm

# captions: a .txt file with one caption per line, or a .tsv / .csv file with a header and a text column
python generate_pseudo_gloss.py --captions captions.tsv --text_column text --out_dir out/ \
    --model Qwen/Qwen3.5-27B --shard 0 --nshards 4
```

Each request is

- system: `prompt.txt` + `", ".join(lexicon keys)`
- user: `Caption: <caption>\nOutput:`

Captions are deduplicated by the md5 of the stripped text and sorted by that hash before sharding. Captions of 600 or more
characters are skipped. The run is resumable. Each output line is

```json
{"h": "<md5 of caption>", "text": "...", "units": [["police", "L"], ["shoot", "L"], ...], "gloss": ["police", "shoot", ...], "ok": 1, "raw": ""}
```

`raw` keeps the LLM reply only when parsing failed (`ok` = 0, empty unit list).

**Post-processing of the reply** (`parse_units.parse_llm_output`). The first `[...]` block of the reply is read as JSON.
Pairs `[unit, tag]` with a tag outside `L W P FS` are dropped, units are lowercased with whitespace collapsed, and empty
units are dropped. A unit tagged `L` that is not a lexicon key is retagged `W`, and a unit tagged `W` that is a lexicon key
is retagged `L`. Any parse error gives an empty unit list.

## Units used in training

```bash
python parse_units.py --gloss out/ --captions captions.tsv --text_column text --id_column clip_id --out units.jsonl
```

A clip gets the `gloss` list of its caption (tags dropped, matched by the md5 of the stripped caption), truncated to the
first 40 units. Clips whose caption has an empty unit list get no pseudo-gloss. When several jsonl files contain the same
caption, the file read last wins (folders are read in sorted file order). These lists are the `pseudo_gloss` field of the
released metadata. In pretraining, the vocabulary is the 30,000 most frequent units plus all lexicon keys.
