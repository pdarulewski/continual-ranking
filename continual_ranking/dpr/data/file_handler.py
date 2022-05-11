import json
import pickle

from torch import Tensor


def read_json_file(path: str) -> list:
    with open(path, mode='r') as f:
        data = json.load(f)
    return data


def save_json_file(data: list, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f)


def pickle_dump(data: list, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(path: str) -> Tensor:
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
