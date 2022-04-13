import collections
import csv
import glob
import logging
from typing import Dict

import torch

from continual_ranking.dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
)

logger = logging.getLogger(__name__)

TableChunk = collections.namedtuple("TableChunk", ["text", "title", "table_id"])


class RetrieverData(torch.utils.data.Dataset):
    def __init__(self, file: str):
        """
        :param file: - real file name or the resource name as they are defined in download_data.py
        """
        self.file = file
        self.data_files = []

    def load_data(self):
        self.data_files = glob.glob(self.file)
        assert (
                len(self.data_files) == 1
        ), "RetrieverData source currently works with single files only. Files specified: {}".format(self.data_files)
        self.file = self.data_files[0]


class CsvCtxSrc(RetrieverData):
    def __init__(
            self,
            file: str,
            id_col: int = 0,
            text_col: int = 1,
            title_col: int = 2,
            id_prefix: str = None,
            normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        logger.info("Reading file %s", self.file)
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                # for row in ifile:
                # row = row.strip().split("\t")
                if row[self.id_col] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col]
                passage = row[self.text_col].strip('"')
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row[self.title_col])
