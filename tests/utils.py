import json
from typing import List

from steamship import Block, File, Tag

from tests import TEST_DATA


def load_config():
    return json.load((TEST_DATA / "config.json").open())


def get_test_file() -> File:
    return File(
        blocks=[
            Block(text="I have cats for pets."),
            Block(text="I have dogs for pets."),
            Block(text="I want a panda for a pet, despite the fact that this is illegal."),
        ]
    )


def get_tag_by_name(tags: List[Tag], name: str) -> Tag:
    for tag in tags:
        if tag.name == name:
            return tag
