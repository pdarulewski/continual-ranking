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

    negatives = []
    for d in tqdm(data):
        records.append({
            'question':      d['question'],
            'positive_ctxs': d['positive_ctxs'][0]['text'],
            'negative_ctxs': [r['text'] for r in d['negative_ctxs'][:6]]
        })
        negatives.extend([i['text'] for i in d['negative_ctxs']])

    del data

    random.shuffle(records)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(['positive_ctxs'])

    train = df.iloc[:15_000].copy()
    val = df.iloc[15_000: 18_000].copy()
    test = df.iloc[18_000: 21_000].copy()
    index = df.iloc[:21_000].copy()

    index = index[['positive_ctxs']]
    index.columns = ['ctxs']
    index = pd.concat([index['ctxs'], pd.Series(negatives)])
    index = pd.DataFrame({'ctxs': index})
    index = index.drop_duplicates()

    for frame in (train, val, test):
        frame['positive_ctxs'] = frame['positive_ctxs'].apply(
            lambda x: [x]
        )

    train.to_json(
        os.path.join(DATA_DIR, 'NQ', 'train.json'),
        orient='records'
    )

    val.to_json(
        os.path.join(DATA_DIR, 'NQ', 'val.json'),
        orient='records'
    )

    test.to_json(
        os.path.join(DATA_DIR, 'NQ', 'test.json'),
        orient='records'
    )

    index.iloc[:500_000].to_json(
        os.path.join(DATA_DIR, 'NQ', 'index.json'),
        orient='records'
    )
