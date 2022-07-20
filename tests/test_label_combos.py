import itertools
from collections import defaultdict

from steamship import Block

from src.api import ZeroShotTaggerPlugin
from tests.utils import load_config


def test_same_results():
    config = load_config()
    config["multi_label"] = True

    labels = sorted({"dog", "groundhog", "tiger", "elephant", "lion", "mouse"})

    label_batches = list(itertools.combinations(labels, 3))
    print(label_batches)
    label_results = defaultdict(list)
    for labels in label_batches:
        config["labels"] = ",".join(labels)
        tagger = ZeroShotTaggerPlugin(config=config)

        block = Block(
            text="I really enjoy seeing large animals in the wild, especially cats and animals with tusks."
        )

        tagger._tag_blocks(blocks=[block], hf_bearer_token=config["hf_api_bearer_token"])
        for tag in block.tags:
            label_results[tag.name].append(tag.value["score"])

    for label in labels:
        assert len(set(label_results[label])) == 1
