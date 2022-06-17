import random
from collections import namedtuple, defaultdict
from typing import List

import torch
from torch.utils.data import Dataset

from continual_ranking.dpr.data.tokenizer import Tokenizer

TrainingSample = namedtuple(
    'TrainingSample', [
        'query',
        'positive_passages',
        'negative_passages'
    ]
)

TokenizedTrainingSample = namedtuple(
    'TokenizedTrainingSample', [
        'question_ids',
        'question_segments',
        'question_attn_mask',
        'context_ids',
        'ctx_segments',
        'ctx_attn_mask',
    ]
)


class TrainTokenizer:
    def __init__(self, max_length: int):
        self.tokenizer = Tokenizer(max_length)

    def __call__(self, sample: TrainingSample) -> TokenizedTrainingSample:
        query_tokens = self.tokenizer(sample.query)

        ctx_tokens = defaultdict(list)

        for ctx in sample.positive_passages + sample.negative_passages:
            tokens = self.tokenizer(ctx)
            ctx_tokens['input_ids'].append(tokens['input_ids'].view(-1))
            ctx_tokens['token_type_ids'].append(tokens['token_type_ids'].view(-1))
            ctx_tokens['attention_mask'].append(tokens['attention_mask'].view(-1))

        return TokenizedTrainingSample(
            query_tokens['input_ids'].view(-1),
            query_tokens['token_type_ids'].view(-1),
            query_tokens['attention_mask'].view(-1),
            torch.stack(ctx_tokens['input_ids']),
            torch.stack(ctx_tokens['token_type_ids']),
            torch.stack(ctx_tokens['attention_mask']),
        )


class TrainDataset(Dataset):

    def __init__(self, data: List[dict], negatives_amount: int, tokenizer: TrainTokenizer):
        self.data = data
        self.negatives_amount = negatives_amount
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> TokenizedTrainingSample:
        json_sample = self.data[idx]

        negatives = self._find_negatives(json_sample)

        sample = TrainingSample(
            json_sample['question'],
            json_sample['positive_ctxs'],
            negatives
        )

        sample = self.tokenizer(sample)

        return sample

    def _find_negatives(self, sample: dict):
        if self.negatives_amount == len(sample['negative_ctxs']):
            return sample['negative_ctxs']

        elif self.negatives_amount > 1:
            negatives = random.sample(self.data, self.negatives_amount)
            negatives = [i['negative_ctxs'][0] for i in negatives]

            current_negative = sample['negative_ctxs']

            if current_negative not in negatives:
                negatives[0] = current_negative[0]

            return negatives

        else:
            return sample['negative_ctxs']
