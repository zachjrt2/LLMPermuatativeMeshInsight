import json, collections, pathlib

directory = 'pairing_results'  # change if needed
files = list(pathlib.Path(directory).glob('*_results.jsonl')) + list(pathlib.Path(directory).glob('*_prompts.jsonl'))
print(f'Found {len(files)} files')

cats = collections.Counter()
for f in files:
    for line in open(f, encoding='utf-8'):
        line = line.strip()
        if not line: continue
        try:
            cats[json.loads(line).get('attack_category', '[missing]')] += 1
        except: pass

for cat, count in cats.most_common():
    print(f'{count:6d}  {repr(cat)}')