from collections import namedtuple
from typing import List

from torch.utils.data import Dataset

from continual_ranking.dpr.data.tokenizer import Tokenizer

IndexSample = namedtuple(
    'IndexSample', [
        'query',
        'positive_passages',
    ]
)

TokenizedIndexSample = namedtuple(
    'TokenizedIndexSample', [
        'input_ids',
        'token_type_ids',
        'attention_mask',
    ]
)


class IndexTokenizer:
    def __init__(self, max_length: int):
        self.tokenizer = Tokenizer(max_length)

    def __call__(self, sample: IndexSample):
        index_tokens = self.tokenizer(sample.positive_passages, sample.query)

        return TokenizedIndexSample(
            index_tokens['input_ids'],
            index_tokens['token_type_ids'],
            index_tokens['attention_mask']
        )


class IndexDataset(Dataset):

    def __init__(self, data: List[dict], tokenizer: IndexTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_sample = self.data[idx]

        sample = IndexSample(
            json_sample['question'],
            json_sample['positive_ctxs'],
        )

        sample = self.tokenizer(sample)

        return sample
