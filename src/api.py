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
        labels: str
        tag_kind: str

    def config_cls(self) -> Type[Config]:
        return self.TaggerPluginConfig

    model_path = "facebook/bart-large-mnli" #"cardiffnlp/twitter-roberta-base-sentiment-latest"


    def make_tags_from_response(self, response, block_len) -> List[Tag.CreateRequest]:
        tags = []
        for i, label in enumerate(response['labels']):
            tags.append(Tag.CreateRequest(kind=self.config.tag_kind,
                                          name= label,
                                          startIdx=0,
                                          endIdx=block_len,
                                          value={'score': response['scores'][i]}))
        return tags

    def tag_blocks(self, blocks : List[Block], hf_bearer_token: str):
        additional_parameters = dict(candidate_labels=self.config.labels.split(','))
        responses = get_huggingface_results(blocks, hf_bearer_token=hf_bearer_token, hf_model_path=self.model_path, additional_params=additional_parameters)
        for i, response in enumerate(responses):
            tags = []
            block = blocks[i]
            tags.extend(self.make_tags_from_response(response, len(block.text)))
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
