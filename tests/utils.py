"""Collection of helper functions to support testing."""
import json
from typing import List

from steamship import Block, File, Tag

from tests import TEST_DATA


def load_config():
    """Load config file from test data."""
    return json.load((TEST_DATA / "config.json").open())


def generate_test_file() -> File:
    """Generate a dummy test File."""
    return File(
        blocks=[
            Block(text="I have cats for pets."),
            Block(text="I have dogs for pets."),
            Block(text="I want a panda for a pet, despite the fact that this is illegal."),
        ]
    )


def get_tag_by_name(tags: List[Tag], name: str) -> Tag:
    """Get based on its name property."""
    for tag in tags:
        if tag.name == name:
            return tag
