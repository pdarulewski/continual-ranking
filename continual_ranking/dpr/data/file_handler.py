import json
import pickle


def read_json_file(path: str) -> list:
    with open(path, mode='r') as f:
        data = json.load(f)
    return data


def save_json_file(data: list, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f)


def store_index(data: list, path: str) -> None:
    with open(path, mode="wb") as f:
        pickle.dump(data, f)
