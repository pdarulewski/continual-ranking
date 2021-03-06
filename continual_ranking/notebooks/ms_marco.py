import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

    df = df.drop_duplicates(['positive_passage'])

    df.columns = ['question', 'positive_ctxs', 'negative_ctxs']
    df = df.sample(frac=1)

    train = df.iloc[:25_000].copy()
    val = df.iloc[25_000: 30_000].copy()
    test = df.iloc[30_000: 35_000].copy()
    index = df.iloc[:350_000].copy()
    index = index[['positive_ctxs', 'negative_ctxs']]
    index = pd.DataFrame({'ctxs': pd.Series(index.values.ravel('F'))})

    for frame in (train, val, test):
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

    val.to_json(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'val.json'),
        orient='records'
    )

    test.to_json(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'test.json'),
        orient='records'
    )

    index.to_json(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'index.json'),
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

    df.to_csv(
        os.path.join(DATA_DIR, 'MSMARCO', 'passages', 'source', 'tokenized_lengths.csv'),
        sep='\t'
    )

    sns.set(font_scale=3, style='whitegrid', rc={"figure.figsize": (16, 11)})

    ax = sns.histplot(df, x='query', bins=60, color='#AC2D38', alpha=0.8)
    ax.set(
        xlabel='Query length',
        ylabel='Frequency',
        title='Query distribution',
        xlim=(0, 50),
        ylim=(0, 15_000),
    )
    plt.savefig(os.path.join(DATA_DIR, 'plot', 'query.pdf'))
    plt.show()

    ax = sns.histplot(df, x='positive_passage', bins=60, color='#AC2D38', alpha=0.8)
    ax.set(
        xlabel='Positive passage length',
        ylabel='Frequency',
        title='Positive passage distribution',
        xlim=(0, 220),
        ylim=(0, 15_000),

    )
    plt.savefig(os.path.join(DATA_DIR, 'plot', 'positives.pdf'))
    plt.show()

    ax = sns.histplot(df, x='negative_passage', bins=60, color='#AC2D38', alpha=0.8)
    ax.set(
        xlabel='Negative passage length',
        ylabel='Frequency',
        title='Negative passage distribution',
        xlim=(0, 220),
        ylim=(0, 15_000),
    )
    plt.savefig(os.path.join(DATA_DIR, 'plot', 'negatives.pdf'))
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
