import json
import re
from pathlib import Path


replacements = re.compile(r'[\n|\t]')

SRC_PATH = Path('./data/abstracts/biorxiv')

headers = [
    'id',
    'source',
    'topic',
    'title',
    'doi',
    'abstract'
]

if __name__ == "__main__":
    json_files = SRC_PATH.glob('*/*.json')

    with (SRC_PATH / 'biorxiv_abstracts.csv').open('w', errors='xmlcharrefreplace') as fp:
        fp.write('\t'.join(headers) + '\n')
        source = 'biorxiv'
        for f in json_files:
            topic = f.parts[-2]
            print(f)
            with f.open('r', errors='xmlcharrefreplace') as json_fp:
                data = json.load(json_fp)
                row = [
                    '{:06d}'.format(int(data['biorxiv_id'])),
                    'biorxiv',
                    topic,
                    replacements.sub(' ', data['title']),
                    data['doi'],
                    '"{}"'.format(replacements.sub(' ', data['abstract']))
                ]
                fp.write('\t'.join(row) + '\n')