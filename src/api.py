"""Example Steamship Tagger Plugin.

In Steamship, **Taggers** are responsible emitting tags that describe the **Steamship Block Format**.
"""

from steamship import Block, Tag, SteamshipError
from steamship.app import App, post, create_handler, Response
from steamship.plugin.tagger import Tagger
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import BlockAndTagPluginOutput
from steamship.plugin.service import PluginRequest
from time import sleep

import json
import requests
import logging
import time


class TaggerPlugin(Tagger, App):
    """"Example Steamship Tagger Plugin."""

    API_URL = "https://api-inference.huggingface.co/models/dslim/bert-large-NER"
    headers = None

    def model_call(self, text: str):
        data = json.dumps(text)
        valid_response = None
        while valid_response is None :
            response = requests.request("POST", self.API_URL, headers=self.headers, data=data)
            json_response = json.loads(response.content.decode("utf-8"))
            logging.info(json_response)
            if 'error' in json_response:
                if not ('is currently loading' in json_response['error']) :
                    raise SteamshipError(message='Unable to query Hugging Face model', internalMessage=f'HF returned error: {json_response["error"]}')
                else:
                    sleep(5)
            else:
                valid_response = json_response
        return valid_response


    def tagBlock(self, block : Block):

        entities = self.model_call(block.text)
        tags = []
        for entity in entities:
            tags.append(Tag.CreateRequest(kind='entity',
                                    name=entity['entity_group'],
                                    startIdx=entity['start'],
                                    endIdx=entity['end'],
                                    value={'score': entity['score']}))

        if block.tags:
            block.tags.extend(tags)
        else:
            block.tags = tags

    def run(self, request: PluginRequest[BlockAndTagPluginInput]) -> Response[BlockAndTagPluginOutput]:
        """Every plugin implements a `run` function.

        This plugin applies sentiment analysis via a pre-trained HF model
        to the text of each Block in a file.
        """

        self.headers = {"Authorization": f"Bearer {self.config.get('hf_api_bearer_token', '')}"}

        if request is None:
            return Response(error=SteamshipError(message="Missing PluginRequest"))

        if request.data is None:
            return Response(error=SteamshipError(message="Missing ParseRequest"))

        if request.data.file is None:
            return Response(error=SteamshipError(message="Missing File"))

        start_time = time.time()
        for block in request.data.file.blocks:
            self.tagBlock(block)
        total_time = time.time() - start_time
        logging.info(f'Completed {len(request.data.file.blocks)} blocks in {total_time} seconds. ({total_time/ len(request.data.file.blocks)} bps)')

        return Response(data=BlockAndTagPluginOutput(request.data.file))

    @post('tag')
    def tag(self, **kwargs) -> dict:
        """App endpoint for our plugin.

        The `run` method above implements the Plugin interface for a Converter.
        This `convert` method exposes it over an HTTP endpoint as a Steamship App.

        When developing your own plugin, you can almost always leave the below code unchanged.
        """
        request = Tagger.parse_request(request=kwargs)
        return self.run(request)


handler = create_handler(TaggerPlugin)
