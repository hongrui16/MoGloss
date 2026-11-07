"""Turn LLM output into pseudo-gloss units, and pseudo-gloss files into the unit lists used in training.

Two steps, both used for the released pseudo-glosses:
  1. parse_llm_output: the raw LLM reply -> [[unit, tag], ...]. Units are lowercased with whitespace collapsed, pairs with an
     unknown tag are dropped, an L unit that is not a lexicon key is retagged W, and a W unit that is a lexicon key is retagged L.
     Any parse error gives an empty list (the caption then has no pseudo-gloss).
  2. training units: the tags are dropped and the first MAX_UNITS (40) units are kept. Captions are matched by the md5 of the
     stripped caption text. Captions with an empty unit list get no pseudo-gloss.

Command line (step 2 for a caption file):
  python parse_units.py --gloss out_dir/ --captions captions.tsv --text_column text --id_column clip_id --out units.jsonl
"""
import argparse, csv, hashlib, json, os, re

TYPES = {'L', 'W', 'P', 'FS'}
MAX_UNITS = 40
csv.field_size_limit(10 ** 9)


def load_lexicon(path):
    """Lexicon keys in file order (first column, header skipped)."""
    keys = []
    with open(path) as f:
        next(f)
        for l in f:
            k = l.rstrip('\n').split('\t')[0]
            if k: keys.append(k)
    return keys


def caption_hash(text):
    return hashlib.md5(text.strip().encode()).hexdigest()


def parse_llm_output(raw, lexicon):
    """raw LLM reply -> [[unit, tag], ...]; lexicon is a set of keys."""
    units = []
    try:
        m = re.search(r'\[.*\]', raw, re.S); js = json.loads(m.group(0)) if m else []
        for x in js:
            if isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and x[1] in TYPES:
                u = re.sub(r'\s+', ' ', x[0].strip().lower()); tg = x[1]
                if not u: continue
                if tg == 'L' and u not in lexicon: tg = 'W'
                elif tg == 'W' and u in lexicon: tg = 'L'
                units.append([u, tg])
    except Exception:
        units = []
    return units


def load_pseudo_glosses(paths):
    """Read pseudo-gloss jsonl files (files, or folders of *.jsonl read in sorted order) -> {caption hash: [unit, ...]}.
    Only non-empty unit lists are kept; a later file overrides an earlier one for the same caption."""
    files = []
    for p in paths:
        if os.path.isdir(p): files += [os.path.join(p, fn) for fn in sorted(os.listdir(p)) if fn.endswith('.jsonl')]
        else: files.append(p)
    table = {}
    for fn in files:
        with open(fn) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l); u = r.get('gloss')
                    if u: table[r['h']] = u
    return table


def training_units(text, table):
    """Units used in training for one caption (tags dropped, at most MAX_UNITS), or None when there are none."""
    u = table.get(caption_hash(text))
    return list(u)[:MAX_UNITS] if u else None


def read_captions(path, text_column='text', id_column=None):
    """Captions as [(id, text)]. A .tsv/.csv file is read with a header row (no quoting, as in the original index files);
    any other file holds one caption per line (id = line number)."""
    out = []
    if path.endswith('.tsv') or path.endswith('.csv'):
        with open(path) as f:
            delim = '\t' if path.endswith('.tsv') else ','
            for i, r in enumerate(csv.DictReader(f, delimiter=delim, quoting=csv.QUOTE_NONE)):
                out.append((r[id_column] if id_column else str(i), r[text_column].strip()))
    else:
        with open(path) as f:
            for i, l in enumerate(f):
                out.append((str(i), l.rstrip('\n').strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description='pseudo-gloss jsonl -> training units per caption')
    ap.add_argument('--gloss', nargs='+', required=True, help='jsonl files or folders written by generate_pseudo_gloss.py')
    ap.add_argument('--captions', required=True, help='.txt (one caption per line) or .tsv/.csv with a header')
    ap.add_argument('--text_column', default='text')
    ap.add_argument('--id_column', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    table = load_pseudo_glosses(a.gloss)
    n = n_ok = 0
    with open(a.out, 'w') as fo:
        for cid, t in read_captions(a.captions, a.text_column, a.id_column):
            if not t: continue
            u = training_units(t, table); n += 1
            if u is None: continue
            fo.write(json.dumps({'id': cid, 'text': t, 'units': u}, ensure_ascii=False) + '\n'); n_ok += 1
    print(f'{n_ok} of {n} captions have pseudo-gloss units -> {a.out}')


if __name__ == '__main__':
    main()
