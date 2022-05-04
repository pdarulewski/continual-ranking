import random
from collections import namedtuple
from typing import List

import torch
from torch.utils.data import Dataset

from continual_ranking.dpr.data.tensorizer import Tensorizer

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


class TrainingDataset(Dataset):

    def __init__(self, data: List[dict], negatives_amount: int):
        self.data = data
        self.negatives_amount = negatives_amount
        self.tokenizer = TrainingTokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_sample = self.data[idx]
        if self.negatives_amount > 1:
            negatives = random.sample(self.data, self.negatives_amount)
            negatives = [i['negative_ctxs'][0] for i in negatives]

            current_negative = json_sample['negative_ctxs']

            if current_negative not in negatives:
                negatives[0] = current_negative[0]

        else:
            negatives = json_sample['negative_ctxs']

        sample = TrainingSample(
            json_sample['question'],
            json_sample['positive_ctxs'],
            negatives
        )

        sample = self.tokenizer(sample)

        return sample


class TrainingTokenizer:
    def __init__(self):
        self.tensorizer = Tensorizer()

    def __call__(self, sample: TrainingSample):
        question_ids = self.tensorizer.text_to_tensor(sample.query)
        question_segments = torch.zeros_like(question_ids)
        question_attn_mask = self.tensorizer.get_attn_mask(question_ids)

        context_ids = [
            self.tensorizer.text_to_tensor(ctx) for ctx in sample.positive_passages + sample.negative_passages
        ]
        context_ids = torch.cat([ctx.view(1, -1) for ctx in context_ids], dim=0)
        ctx_segments = torch.zeros_like(context_ids)
        ctx_attn_mask = self.tensorizer.get_attn_mask(context_ids)

        return TokenizedTrainingSample(
            question_ids,
            question_segments,
            question_attn_mask,
            context_ids,
            ctx_segments,
            ctx_attn_mask
        )
