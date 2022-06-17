import json
import os

import pandas as pd
from tqdm import tqdm

from continual_ranking.paths import DATA_DIR

MS_MARCO_PATH = os.path.join(DATA_DIR, 'MSMARCO', 'documents')


def index():
    dfs = pd.read_csv(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'source', 'triples.train.small.tsv.gz'),
        chunksize=10_000, sep='\t', header=None, usecols=[1, 2],
        names=['positive_passage', 'negative_passage']
    )

    counter = 0
    p_bar = tqdm(dfs, desc='processed = 0.00%')
    for df in p_bar:
        tmp = pd.Series(df.values.ravel('F'))
        tmp.to_csv(
            os.path.join(DATA_DIR, 'index.tsv.gz'),
            sep='\t',
            index=False,
            mode='a',
            header=False
        )
        counter += 10_000
        p_bar.set_description(f'processed = {(counter / 39_780_811) * 100: .2f}%')

    for file_name in ['nq-train.json', 'nq-dev.json']:
        ctxs = []
        with open(os.path.join(DATA_DIR, 'NQ', 'source', file_name)) as file:
            data = json.load(file)

        counter = 0
        data_length = len(data)
        p_bar = tqdm(data, desc='processed = 0.00%')
        for d in p_bar:
            ctxs.extend([i['text'] for i in d['positive_ctxs']])
            ctxs.extend([i['text'] for i in d['negative_ctxs']])

            counter += 1
            p_bar.set_description(f'processed = {(counter / data_length) * 100: .2f}%')

        del data

        df = pd.DataFrame({'ctxs': ctxs})
        df.to_csv(
            os.path.join(DATA_DIR, 'index.tsv.gz'),
            sep='\t',
            index=False,
            mode='a',
            header=False
        )
