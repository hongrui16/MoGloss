"""Generate pseudo-glosses from English captions with a local LLM (vLLM, greedy decoding).

Each unique caption is sent once. The system prompt is prompt.txt followed by the lexicon keys joined with ", ", and the user
message is "Caption: <caption>\\nOutput:". The reply is parsed by parse_units.parse_llm_output.

Output: <out_dir>/pseudo_gloss_shard<k>.jsonl, one line per caption
  {"h": md5 of the caption, "text": caption, "units": [[unit, tag], ...], "gloss": [unit, ...], "ok": 0/1, "raw": reply if not ok}
The run is resumable (captions already in the output file are skipped). Captions of 600 characters or more are skipped.

  python generate_pseudo_gloss.py --captions captions.tsv --text_column text --out_dir out/ --shard 0 --nshards 4
"""
import argparse, json, os, sys

_here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _here)
from parse_units import TYPES, load_lexicon, caption_hash, parse_llm_output, read_captions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', required=True, help='.txt (one caption per line) or .tsv/.csv with a header')
    ap.add_argument('--text_column', default='text')
    ap.add_argument('--lexicon', default=os.path.join(_here, 'sign_lexicon.tsv'))
    ap.add_argument('--prompt', default=os.path.join(_here, 'prompt.txt'))
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3.5-27B')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--tp', type=int, default=1, help='vLLM tensor parallel size')
    ap.add_argument('--max_items', type=int, default=0)
    ap.add_argument('--batch', type=int, default=2000)
    ap.add_argument('--max_model_len', type=int, default=24576)
    ap.add_argument('--max_chars', type=int, default=600, help='skip captions with this many characters or more')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

    keys = load_lexicon(a.lexicon); kset = set(keys)
    with open(a.prompt) as f: system = f.read() + ', '.join(keys)

    # unique captions, sorted by hash, then sharded
    texts = {}
    for _, t in read_captions(a.captions, a.text_column):
        if t: texts.setdefault(caption_hash(t), t)
    items = sorted(texts.items())[a.shard::a.nshards]
    out = os.path.join(a.out_dir, f'pseudo_gloss_shard{a.shard}.jsonl'); done = set()
    if os.path.exists(out):
        with open(out) as f: done = {json.loads(l)['h'] for l in f if l.strip()}
    items = [(h, t) for h, t in items if h not in done and len(t) < a.max_chars]
    if a.max_items: items = items[:a.max_items]
    print(f'shard {a.shard}: {len(items)} captions to do ({len(done)} done), lexicon {len(keys)} keys', flush=True)
    if not items: print(a.shard, 'DONE 0'); return

    from vllm import LLM, SamplingParams
    llm = LLM(a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, enable_prefix_caching=True,
              gpu_memory_utilization=0.92, dtype='bfloat16', limit_mm_per_prompt={'image': 0, 'video': 0})
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    n = n_ok = 0; stats = {t: 0 for t in TYPES}
    with open(out, 'a') as fo:
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            msgs = [[{'role': 'system', 'content': system}, {'role': 'user', 'content': f'Caption: {t}\nOutput:'}] for _, t in chunk]
            res = llm.chat(msgs, sp, chat_template_kwargs={'enable_thinking': False}, use_tqdm=False)
            for (h, t), r in zip(chunk, res):
                raw = r.outputs[0].text.strip(); units = parse_llm_output(raw, kset)
                ok = int(bool(units))
                for _, tg in units: stats[tg] += 1
                fo.write(json.dumps({'h': h, 'text': t, 'units': units, 'gloss': [u for u, _ in units], 'ok': ok,
                                     'raw': raw if not ok else ''}, ensure_ascii=False) + '\n')
                n += 1; n_ok += ok
            fo.flush(); tot = max(1, sum(stats.values()))
            print(f'{a.shard} {n}/{len(items)} ok {n_ok / max(n, 1):.3f} | ' + ' '.join(f'{k} {v / tot:.3f}' for k, v in sorted(stats.items())), flush=True)
    print(a.shard, 'DONE', n, f'ok_rate {n_ok / max(n, 1):.3f}', flush=True)


if __name__ == '__main__':
    main()
