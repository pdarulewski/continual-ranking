import itertools
import json
import logging
import math
import random
from typing import List, Iterator, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor as T

logger = logging.getLogger()


def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % path)
            data = json.load(f)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question


def pad_to_len(seq: T, pad_id: int, max_len: int):
    s_len = seq.size(0)
    if s_len > max_len:
        return seq[0:max_len]
    return torch.cat([seq, torch.Tensor().new_full((max_len - s_len,), pad_id, dtype=torch.long)], dim=0)


class Tensorizer:
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(
            self,
            text: str,
            title: str = None,
            add_special_tokens: bool = True,
            apply_max_len: bool = True,
    ):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError


class RepTokenSelector:
    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        raise NotImplementedError


class RepStaticPosTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        self.static_position = static_position

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        return self.static_position


DEFAULT_SELECTOR = RepStaticPosTokenSelector()


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            selector: DictConfig = None,
            special_token: str = None,
            shuffle_positives: bool = False,
            query_special_suffix: str = None,
            encoder_type: str = None,
    ):
        if selector:
            self.selector = hydra.utils.instantiate(selector)
        else:
            self.selector = DEFAULT_SELECTOR
        self.special_token = special_token
        self.encoder_type = encoder_type
        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix
        self.data = []

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        raise NotImplementedError

    def calc_total_data_len(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError

    def _process_query(self, query: str):
        # as of now, always normalize query
        query = normalize_question(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix

        return query


# TODO: to be fully replaced with LocalSharded{...}. Keeping it only for old results reproduction compatibility
class ShardedDataIterator:
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """

    def __init__(
            self,
            dataset: Dataset,
            shard_id: int = 0,
            num_shards: int = 1,
            batch_size: int = 1,
            shuffle=True,
            shuffle_seed: int = 0,
            offset: int = 0,
            strict_batch_size: bool = False,
    ):

        self.dataset = dataset
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size
        self.shard_start_idx = -1
        self.shard_end_idx = -1
        self.max_iterations = 0

    def calculate_shards(self):
        logger.info("Calculating shard positions")
        shards_num = max(self.num_shards, 1)
        shard_id = max(self.shard_id, 0)

        total_size = self.dataset.calc_total_data_len()
        samples_per_shard = math.ceil(total_size / shards_num)

        self.shard_start_idx = shard_id * samples_per_shard
        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        if self.strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / self.batch_size)
        else:
            self.max_iterations = int(samples_per_shard / self.batch_size)

        logger.info(
            "samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d",
            samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations,
        )

    def load_data(self):
        self.calculate_shards()
        self.dataset.load_data()
        logger.info("Sharded dataset data %d", len(self.dataset))

    def total_data_len(self) -> int:
        return len(self.dataset)

    def iterations_num(self) -> int:
        return self.max_iterations - self.iteration

    def max_iterations_num(self) -> int:
        return self.max_iterations

    def get_iteration(self) -> int:
        return self.iteration

    def get_shard_indices(self, epoch: int):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices[self.shard_start_idx: self.shard_end_idx]
        return shard_indices

    def iterate_ds_sampled_data(self, num_iterations: int, epoch: int = 0) -> Iterator[List]:
        self.iteration = 0
        shard_indices = self.get_shard_indices(epoch)
        cycle_it = itertools.cycle(shard_indices)
        for i in range(num_iterations):
            items_idxs = [next(cycle_it) for _ in range(self.batch_size)]
            self.iteration += 1
            items = [self.dataset[idx] for idx in items_idxs]
            yield items

        logger.info("Finished iterating, iteration={}, shard={}".format(self.iteration, self.shard_id))
        # TODO: reset the iteration status?
        self.iteration = 0


class MultiSetDataIterator:
    """
    Iterator over multiple data sources. Useful when all samples form a single batch should be from the same dataset.
    """

    def __init__(
            self,
            datasets: List[ShardedDataIterator],
            shuffle_seed: int = 0,
            shuffle=True,
            sampling_rates: List = [],
            rank: int = 0,
    ):
        # randomized data loading to avoid file system congestion
        ds_list_copy = [ds for ds in datasets]
        rnd = random.Random(rank)
        rnd.shuffle(ds_list_copy)
        [ds.load_data() for ds in ds_list_copy]

        self.iterables = datasets
        data_lengths = [it.total_data_len() for it in datasets]
        self.total_data = sum(data_lengths)
        logger.info("rank=%d; Multi set data sizes %s", rank, data_lengths)
        logger.info("rank=%d; Multi set total data %s", rank, self.total_data)
        logger.info("rank=%d; Multi set sampling_rates %s", rank, sampling_rates)
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.iteration = 0
        self.rank = rank

        if sampling_rates:
            self.max_its_pr_ds = [int(ds.max_iterations_num() * sampling_rates[i]) for i, ds in enumerate(datasets)]
        else:
            self.max_its_pr_ds = [ds.max_iterations_num() for ds in datasets]

        self.max_iterations = sum(self.max_its_pr_ds)
        logger.info("rank=%d; Multi set max_iterations per dataset %s", rank, self.max_its_pr_ds)
        logger.info("rank=%d; Multi set max_iterations %d", rank, self.max_iterations)

    def total_data_len(self) -> int:
        return self.total_data

    def get_max_iterations(self):
        return self.max_iterations

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[Tuple[List, int]]:

        logger.info("rank=%d; Iteration start", self.rank)
        logger.info(
            "rank=%d; Multi set iteration: iteration ptr per set: %s",
            self.rank,
            [it.get_iteration() for it in self.iterables],
        )

        data_src_indices = []
        iterators = []
        for source, src_its in enumerate(self.max_its_pr_ds):
            logger.info(
                "rank=%d; Multi set iteration: source %d, batches to be taken: %s",
                self.rank,
                source,
                src_its,
            )
            data_src_indices.extend([source] * src_its)

            iterators.append(self.iterables[source].iterate_ds_sampled_data(src_its, epoch=epoch))

        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(data_src_indices)

        logger.info("rank=%d; data_src_indices len=%d", self.rank, len(data_src_indices))
        for i, source_idx in enumerate(data_src_indices):
            it = iterators[source_idx]
            next_item = next(it, None)
            if next_item is not None:
                self.iteration += 1
                yield (next_item, source_idx)
            else:
                logger.warning("rank=%d; Next item in the source %s is None", self.rank, source_idx)

        logger.info("rank=%d; last iteration %d", self.rank, self.iteration)

        logger.info(
            "rank=%d; Multi set iteration finished: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables],
        )
        [next(it, None) for it in iterators]

        # TODO: clear iterators in some non-hacky way
        for it in self.iterables:
            it.iteration = 0
        logger.info(
            "rank=%d; Multi set iteration finished after next: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables],
        )
        # reset the iteration status
        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration
