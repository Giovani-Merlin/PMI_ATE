import json


def get_dataset(type_: str = "train"):
    with open("data/laptops_" + type_ + ".json", "r") as f:
        return json.load(f)
