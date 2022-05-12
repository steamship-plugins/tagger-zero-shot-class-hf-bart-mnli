from steamship.data.parser import ParseRequest
from steamship.plugin.service import PluginRequest
from steamship import BlockTypes
from src.api import ParserPlugin
import os
from typing import List

__copyright__ = "Steamship"
__license__ = "MIT"

def _read_test_file_lines(filename: str) -> List[str]:
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', filename), 'r') as f:
        return f.read().split('\n')

def test_parser():
    parser = ParserPlugin()
    rose_lines = _read_test_file_lines('roses.txt')
    request = PluginRequest(data=ParseRequest(
        docs=rose_lines
    ))
    response = parser.run(request)

    assert(response.error is None)
    assert(response.data is not None)

    assert (response.data.blocks is not None)
    assert (len(response.data.blocks) == 3)

    # A Poem
    para1 = response.data.blocks[0]
    assert(para1.type == BlockTypes.Document)
    assert(para1.children is not None)
    for kit in para1.children:
        print(kit.text)
    assert(len(para1.children) == 1)
    assert(para1.children[0].text == "A Poem.")
    assert(para1.children[0].tokens is not None)
    assert(len(para1.children[0].tokens) == 2)
    assert(para1.children[0].type == BlockTypes.Sentence)

    # Roses are red. Violets are blue.
    para2 = response.data.blocks[1]
    assert(para2.type == BlockTypes.Document)
    assert(para2.children is not None)
    assert(len(para2.children) == 2)
    assert(para2.children[0].text == "Roses are red.")
    assert(para2.children[1].text == "Violets are blue.")
    assert(para2.children[1].tokens[1].text == "are")

    # Sugar is sweet, and I love you.
    para3 = response.data.blocks[2]
    assert(para3.children is not None)
    assert(len(para3.children) == 1)
    assert(para3.children[0].text == "Sugar is sweet, and I love you.")
    assert(para3.children[0].tokens[2].text == "sweet,")
