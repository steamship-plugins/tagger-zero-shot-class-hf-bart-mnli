import json
import os

from steamship import Block

from src.api import ZeroShotTaggerPlugin


def test_same_results():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "..", "test_data", "tests-config.json"), "r") as config_file:
        config = json.load(config_file)
    config["multi_label"] = False

    text = (
        "I really enjoy seeing large animals in the wild, especially cats and animals with tusks."
    )
    labels = ["dog", "groundhog", "tiger", "elephant", "lion", "mouse"]

    labelsets = [[0, 1, 2], [3, 4, 5], [2, 4, 5], [1, 3, 4], [0, 2, 3], [5, 0, 1]]

    results = []
    for labelset in labelsets:
        this_call_labels = [label for i, label in enumerate(labels) if i in labelset]
        config["labels"] = ",".join(this_call_labels)
        tagger = ZeroShotTaggerPlugin(config=config)
        block = Block(text=text)
        tagger._tag_blocks([block], config["hf_api_bearer_token"])
        this_call_results = {tag.name: tag.value["score"] for tag in block.tags}
        results.append(this_call_results)

    label_results = {
        label: [result[label] for result in results if label in result] for label in labels
    }
    print(label_results)
    for label in labels:
        assert len(set(label_results[label])) == 1
        print(f"{label}: {label_results[label][0]}")
