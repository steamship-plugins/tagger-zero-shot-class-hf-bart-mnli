import pytest
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.service import PluginRequest

from src.api import ZeroShotTaggerPlugin
from tests.utils import get_tag_by_name, get_test_file, load_config


@pytest.mark.parametrize("multi_label", [True, False])
def test_parser(multi_label: bool):
    config = load_config()
    config["multi_label"] = multi_label

    tagger = ZeroShotTaggerPlugin(config=config)
    test_file = get_test_file()
    request = PluginRequest(data=BlockAndTagPluginInput(file=test_file))
    response = tagger.run(request)

    assert response.data is not None

    assert response.data.file is not None
    assert len(response.data.file.blocks) == 3
    if multi_label:
        assert len(response.data.file.blocks[0].tags) == 3
    else:
        assert len(response.data.file.blocks[0].tags) == 1

    for i, animal_class in enumerate(("cats", "dogs", "pandas")):
        tag = get_tag_by_name(response.data.file.blocks[i].tags, animal_class)
        assert tag.kind == "my-animal-classification"
        assert tag.value["score"] > 0.9


def test_many_labels():
    config = load_config()
    config["multi_label"] = True
    config["labels"] = "one,two,three,four,five,six,seven,eight,nine,ten,eleven"

    tagger = ZeroShotTaggerPlugin(config=config)
    test_file = get_test_file()
    request = PluginRequest(data=BlockAndTagPluginInput(file=test_file))
    response = tagger.run(request)

    assert response.data is not None
    assert len(response.data.file.blocks) == 3
    for block in response.data.file.blocks:
        assert len(block.tags) == 11
