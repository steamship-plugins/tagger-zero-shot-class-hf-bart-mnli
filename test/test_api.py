from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.service import PluginRequest
from steamship import File, Block
from src.api import TaggerPlugin
import os
import json

__copyright__ = "Steamship"
__license__ = "MIT"

def _get_test_file() -> File:
    return File(blocks=[
        Block(text='My name is Dave and I live near Baltimore'),
        Block(text='My name is Ted and I live in Washington, DC'),
        Block(text='My name is Enias and I live in Brussels'),
    ])

def test_parser():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'test-config.json'), 'r') as config_file:
        config = json.load(config_file)

    tagger = TaggerPlugin(config=config)
    test_file = _get_test_file()
    request = PluginRequest(data=BlockAndTagPluginInput(
        file=test_file
    ))
    response = tagger.run(request)

    assert(response.data is not None)

    assert (response.data.file is not None)
    assert (len(response.data.file.blocks) == 3)
    assert (len(response.data.file.blocks[0].tags) == 2)
    tag0 = response.data.file.blocks[0].tags[0]
    assert(tag0.kind == 'entity')
    assert(tag0.name == 'PER')
    assert(tag0.startIdx == 11)
    assert(tag0.endIdx == 15)
    assert(tag0.value['score'] > .9)
    tag1 = response.data.file.blocks[0].tags[1]
    assert (tag1.kind == 'entity')
    assert (tag1.name == 'LOC')
    assert (tag1.startIdx == 32)
    assert (tag1.endIdx == 41)
    assert (tag1.value['score'] > .9)
