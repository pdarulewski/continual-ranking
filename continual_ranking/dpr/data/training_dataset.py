import json
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

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


def read_json_file(path: str) -> dict:
    with open(path, mode='r') as f:
        data = json.load(f)
    return data


class TrainingDataset(Dataset):

    def __init__(self, file: str):
        self.data = read_json_file(file)
        self.transform = transforms.Compose([
            Tokenize()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_sample = self.data[idx]
        sample = TrainingSample(
            json_sample['question'],
            json_sample['positive_ctxs'],
            json_sample['negative_ctxs']
        )

        if self.transform:
            sample = self.transform(sample)

        return sample


class Tokenize:
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
