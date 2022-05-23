"""Example Steamship Tagger Plugin.

In Steamship, **Taggers** are responsible emitting tags that describe the **Steamship Block Format**.
"""

from steamship import Block, Tag, SteamshipError
from steamship.app import App, post, create_handler, Response
from steamship.plugin.tagger import Tagger
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import BlockAndTagPluginOutput
from steamship.plugin.service import PluginRequest
from steamship.plugin.config import Config
from steamship.utils.huggingface_helper import get_huggingface_results

import logging
import time
from typing import List, Type

class TaggerPlugin(Tagger, App):
    """Example Steamship Tagger Plugin."""


    class TaggerPluginConfig(Config):
        hf_api_bearer_token: str

    def config_cls(self) -> Type[Config]:
        return self.TaggerPluginConfig

    model_path = "dslim/bert-large-NER"

    def make_tags_from_response(self, response) -> List[Tag.CreateRequest]:
        tags = []
        for entity in response:
            tags.append(Tag.CreateRequest(kind='entity',
                                          name=entity['entity_group'],
                                          startIdx=entity['start'],
                                          endIdx=entity['end'],
                                          value={'score': entity['score']}))
        return tags

    def tag_blocks(self, blocks : List[Block], hf_bearer_token: str):
        responses = get_huggingface_results(blocks, hf_bearer_token=hf_bearer_token, hf_model_path=self.model_path)
        for i, response in enumerate(responses):
            tags = []
            tags.extend(self.make_tags_from_response(response))
            block = blocks[i]
            if block.tags:
                block.tags.extend(tags)
            else:
                block.tags = tags

    def run(self, request: PluginRequest[BlockAndTagPluginInput]) -> Response[BlockAndTagPluginOutput]:
        """Every plugin implements a `run` function.

        This plugin applies sentiment analysis via a pre-trained HF model
        to the text of each Block in a file.
        """
        logging.info('Invoking tagger-entity-hf-bert')
        if request is None:
            return Response(error=SteamshipError(message="Missing PluginRequest"))

        if request.data is None:
            return Response(error=SteamshipError(message="Missing ParseRequest"))

        if request.data.file is None:
            return Response(error=SteamshipError(message="Missing File"))

        self.tag_blocks(request.data.file.blocks, self.config.hf_api_bearer_token)

        return Response(data=BlockAndTagPluginOutput(file=request.data.file))


handler = create_handler(TaggerPlugin)
