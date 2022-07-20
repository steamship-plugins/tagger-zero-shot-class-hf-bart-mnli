import json
import os
from typing import List

from steamship import Block, File, Tag
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.service import PluginRequest

from src.api import ZeroShotTaggerPlugin


def _get_test_file() -> File:
    return File(
        blocks=[
            Block(text="I have cats for pets."),
            Block(text="I have dogs for pets."),
            Block(text="I want a panda for a pet, despite the fact that this is illegal."),
        ]
    )


def _get_tag_by_name(tags: List[Tag], name: str) -> Tag:
    for tag in tags:
        if tag.name == name:
            return tag
    return None


def test_parser():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "..", "test_data", "tests-config.json"), "r") as config_file:
        config = json.load(config_file)

    tagger = ZeroShotTaggerPlugin(config=config)
    test_file = _get_test_file()
    request = PluginRequest(data=BlockAndTagPluginInput(file=test_file))
    response = tagger.run(request)

    assert response.data is not None

    assert response.data.file is not None
    assert len(response.data.file.blocks) == 3
    assert len(response.data.file.blocks[0].tags) == 3
    cat_tag = _get_tag_by_name(response.data.file.blocks[0].tags, "cats")
    assert cat_tag.kind == "my-animal-classification"
    assert cat_tag.value["score"] > 0.9
    dog_tag = _get_tag_by_name(response.data.file.blocks[1].tags, "dogs")
    assert dog_tag.kind == "my-animal-classification"
    assert dog_tag.value["score"] > 0.9
    panda_tag = _get_tag_by_name(response.data.file.blocks[2].tags, "pandas")
    assert panda_tag.kind == "my-animal-classification"
    assert panda_tag.value["score"] > 0.9


def test_many_labels():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "..", "test_data", "tests-config.json"), "r") as config_file:
        config = json.load(config_file)

    config["multi_label"] = True
    config["labels"] = "one,two,three,four,five,six,seven,eight,nine,ten,eleven"
    tagger = ZeroShotTaggerPlugin(config=config)
    test_file = _get_test_file()
    request = PluginRequest(data=BlockAndTagPluginInput(file=test_file))
    response = tagger.run(request)

    assert response.data is not None
    assert len(response.data.file.blocks) == 3
    for block in response.data.file.blocks:
        assert len(block.tags) == 11
