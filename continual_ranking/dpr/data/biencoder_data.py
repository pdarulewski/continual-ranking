import collections
import glob
import logging
import os
from typing import List

from omegaconf import DictConfig

from continual_ranking.config.paths import DATA_DIR
from continual_ranking.dpr.utils.data_utils import read_data_from_json_files, Dataset

logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])


class BiEncoderSample:
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class JsonQADataset(Dataset):
    def __init__(
            self,
            file: str,
            selector: DictConfig = None,
            special_token: str = None,
            encoder_type: str = None,
            shuffle_positives: bool = False,
            normalize: bool = False,
            query_special_suffix: str = None,
            # tmp: for cc-net results only
            exclude_gold: bool = False,
    ):
        super().__init__(
            selector,
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        self.file = file
        self.data_files = []
        self.normalize = normalize
        self.exclude_gold = exclude_gold

    def calc_total_data_len(self):
        if not self.data:
            logger.info("Loading all data")
            self._load_all_data()
        return len(self.data)

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        if not self.data:
            self._load_all_data()
        if start_pos >= 0 and end_pos >= 0:
            logger.info("Selecting subset range from %d to %d", start_pos, end_pos)
            self.data = self.data[start_pos:end_pos]

    def _load_all_data(self):
        self.data_files = glob.glob(os.path.join(DATA_DIR, self.file))
        logger.info("Data files: %s", self.data_files)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: %d", len(self.data))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = self._process_query(json_sample["question"])

        positive_ctxs = json_sample["positive_ctxs"]
        if self.exclude_gold:
            ctxs = [ctx for ctx in positive_ctxs if "score" in ctx]
            if ctxs:
                positive_ctxs = ctxs

        negative_ctxs = json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        hard_negative_ctxs = json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else []

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        return r


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text
