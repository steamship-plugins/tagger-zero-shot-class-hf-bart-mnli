"""Example Steamship Tagger Plugin.

In Steamship, **Taggers** are responsible emitting tags that describe the **Steamship Block Format**.
"""

from steamship import Block, BlockTypes, MimeTypes, SteamshipError
from steamship.app import App, post, create_handler, Response
from steamship.plugin.parser import Parser
from steamship.data.parser import ParseResponse, ParseRequest
from steamship.plugin.service import PluginResponse, PluginRequest
from steamship import Token, Block

def _makeSentenceBlock(sentence: str, includeTokens: bool = True) -> Block:
    """Splits the sentence in tokens on space. Dep head of all is first token"""
    if includeTokens:
        tokens = [Token(text=word, parentIndex=0) for word in sentence.split(" ")]
        return Block(type=BlockTypes.Sentence, text=sentence, tokens=tokens)
    else:
        return Block(type=BlockTypes.Sentence, text=sentence)


def _makeDocBlock(text: str, includeTokens=True) -> Block:
    """Splits the document into sentences by assuming a period is a sentence divider."""
    # Add the period back
    sentences = map(lambda x: x.strip(), text.split("."))
    sentences = list(filter(lambda s: len(s) > 0, sentences))
    sentences = list(map(lambda s: "{}.".format(s), sentences))
    children = [_makeSentenceBlock(sentence, includeTokens=includeTokens) for sentence in sentences]
    return Block(text=text, type=BlockTypes.Document, children=children)


class TaggerPlugin(Parser, App):
    """"Example Steamship Tagger Plugin."""

    def run(self, request: PluginRequest[ParseRequest]) -> PluginResponse[ParseResponse]:
        """Every plugin implements a `run` function.

        This template plugin does an extremely simple form of text parsing:
            - It chunks the incoming data into sentences on ANY period that it sees.
            - It chunks each sentence into tokens on any whitespace it sees.
            - It assigns the first token in a sentence to be the head.
        """
        if request is None:
            return Response(error=SteamshipError(message="Missing PluginRequest"))

        if request.data is None:
            return Response(error=SteamshipError(message="Missing ParseRequest"))

        if request.data.docs is None:
            return Response(error=SteamshipError(message="Missing `docs` field in ParseRequest"))

        blocks = list(map(
            lambda text: _makeDocBlock(text, includeTokens = request.data.includeTokens),
            request.data.docs
        ))

        return PluginResponse(data=ParseResponse(blocks=blocks))

    @post('tag')
    def tag(self, **kwargs) -> Response:
        """App endpoint for our plugin.

        The `run` method above implements the Plugin interface for a Converter.
        This `convert` method exposes it over an HTTP endpoint as a Steamship App.

        When developing your own plugin, you can almost always leave the below code unchanged.
        """
        request = Parser.parse_request(request=kwargs)
        response = self.run(request)
        dict_response = Parser.response_to_dict(response)
        return Response(json=dict_response)


handler = create_handler(TaggerPlugin)
