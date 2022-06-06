import json
import os
import random

import pandas as pd
from tqdm import tqdm

from continual_ranking.paths import DATA_DIR


def nq() -> None:
    records = []
    with open(os.path.join(DATA_DIR, 'NQ', 'source', 'nq-train.json')) as file:
        data = json.load(file)

    for d in tqdm(data):
        records.append({
            'question':      d['question'],
            'positive_ctxs': d['positive_ctxs'][0]['text'],
            'negative_ctxs': [r['text'] for r in d['negative_ctxs'][:6]]
        })

    del data

    random.shuffle(records)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(['positive_ctxs'])

    train = df.iloc[:15_000].copy()
    val = df.iloc[15_000: 18_000].copy()
    test = df.iloc[18_000: 21_000].copy()
    index = df.iloc[:21_000].copy()
    index = index[['question', 'positive_ctxs']]

    for frame in (train, val, test):
        frame['positive_ctxs'] = frame['positive_ctxs'].apply(
            lambda x: [x]
        )
        frame['negative_ctxs'] = frame['negative_ctxs'].apply(
            lambda x: [x]
        )

    train.to_json(
        os.path.join(DATA_DIR, 'NQ', 'train_0.json'),
        orient='records'
    )

    val.to_json(
        os.path.join(DATA_DIR, 'NQ', 'val_0.json'),
        orient='records'
    )

    test.to_json(
        os.path.join(DATA_DIR, 'NQ', 'test_0.json'),
        orient='records'
    )

    index.to_json(
        os.path.join(DATA_DIR, 'NQ', 'index_0.json'),
        orient='records'
    )
