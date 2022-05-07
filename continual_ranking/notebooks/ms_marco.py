import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from continual_ranking.paths import DATA_DIR

MS_MARCO_PATH = os.path.join(DATA_DIR, 'MSMARCO', 'documents')


def wiki():
    dfs = pd.read_csv(
        os.path.join(DATA_DIR, 'MSMARCO', 'documents', 'wikipedia.tsv.gz'),
        chunksize=10_000, sep='\t', header=None, usecols=[1, 2]
    )

    counter = 0
    for df in tqdm(dfs):
        counter += len(df)

        df[2] = df[2].str.replace(r'From Wikipedia, the free encyclopedia', '')
        df[2] = df[2].str.replace(r'\[.*?\]', '')

        # df.to_json(os.path.join(MS_MARCO_PATH, 'wiki_clean.json'), orient='records', lines=True)
        df.to_csv(os.path.join(MS_MARCO_PATH, 'wiki_clean.tsv.gz'), index=False, mode='a')


def wiki_triplets():
    dfs = pd.read_csv(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'triples.train.small.tsv.gz'),
        chunksize=100_000, sep='\t', header=None,
        names=['query', 'positive_passage', 'negative_passage']
    )

    counter = 0
    np.random.seed(42)
    for df in tqdm(dfs):
        # counter += len(df)
        # 39_780_811 -> 398 chunks
        df.sample(200).to_csv(
            os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'subset.tsv.gz'),
            sep='\t',
            index=False,
            mode='a'
        )


def wiki_parsed():
    df = pd.read_csv(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'source', 'subset.tsv.gz'),
        sep='\t'
    )

    df.columns = ['question', 'positive_ctxs', 'negative_ctxs']

    embeddings = df.copy(True)

    train, dev, test = np.split(df.sample(frac=1, random_state=42), [int(.6 * len(df)), int(.8 * len(df))])

    for frame in (train, dev):
        frame['positive_ctxs'] = frame['positive_ctxs'].apply(
            lambda x: [x]
        )
        frame['negative_ctxs'] = frame['negative_ctxs'].apply(
            lambda x: [x]
        )

    train.to_json(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'train.json'),
        orient='records'
    )

    dev.to_json(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'eval.json'),
        orient='records'
    )

    test = test[['question', 'positive_ctxs']]
    test['positive_ctxs'] = test['positive_ctxs'].apply(
        lambda x: [x]
    )

    test.to_json(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'test.json'),
        orient='records'
    )

    embeddings = embeddings[['question', 'positive_ctxs']]
    embeddings.to_json(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'embeddings.json'),
        orient='records'
    )


def lengths():
    df = pd.read_csv(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'source', 'subset.tsv.gz'),
        sep='\t'
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: len(tokenizer.encode(x))
        )
        print(f'{col} done!')

    plt.hist(df['query'], bins=100)
    plt.show()

    plt.hist(df['positive_passage'], bins=100)
    plt.show()

    plt.hist(df['negative_passage'], bins=100)
    plt.show()


def main():
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(
        'Hello, my dog is cute',
        'Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute',
        return_tensors='pt',
        max_length=512,
        padding='max_length',
    )


if __name__ == '__main__':
    main()
