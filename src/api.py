"""Example Steamship Tagger Plugin.

In Steamship, **Taggers** are responsible emitting tags that describe the **Steamship Block Format**.
"""

from steamship import Block, Tag, SteamshipError
from steamship.app import App, post, create_handler, Response
from steamship.plugin.tagger import Tagger
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import BlockAndTagPluginOutput
from steamship.plugin.service import PluginRequest

import logging
import time
from typing import List
import websockets
import json
import asyncio
import uuid

async def send(websocket, payloads, hf_token):
    # You need to login with a first message as headers are not forwarded
    # for websockets
    await websocket.send(f"Bearer {hf_token}".encode("utf-8"))
    for payload in payloads:
        await websocket.send(json.dumps(payload).encode("utf-8"))

async def recv(websocket, last_id):
    outputs = []
    while True:
        data = await websocket.recv()
        payload = json.loads(data)
        if payload["type"] == "results":
            # {"type": "results", "outputs": JSONFormatted results, "id": the id we sent}
            outputs.append(payload["outputs"])
            if payload["id"] == last_id:
                return outputs
        else:
            # {"type": "status", "message": "Some information about the queue"}
            logging.info(f"HF Status message: {payload['message']}")
            pass

async def call_model_async(block_texts: List[str], hf_bearer_token : str, hf_model_path: str, hf_compute_type: str = 'cpu'):

    uri = f"wss://api-inference.huggingface.co/bulk/stream/{hf_compute_type}/{hf_model_path}"
    async with websockets.connect(uri) as websocket:
        # inputs and parameters are classic, "id" is a way to track that query
        payloads = [
            {
                "id": str(uuid.uuid4()),
                "inputs": text
            }
            for i, text in enumerate(block_texts)
        ]
        last_id = payloads[-1]["id"]
        future = send(websocket, payloads, hf_bearer_token)
        future_r = recv(websocket, last_id)
        _, outputs = await asyncio.gather(future, future_r)
    return outputs

def call_model(blocks: List[Block], hf_bearer_token : str, hf_model_path: str, hf_compute_type: str = 'cpu'):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(call_model_async([block.text for block in blocks], hf_bearer_token, hf_model_path, hf_compute_type))

class TaggerPlugin(Tagger, App):
    """"Example Steamship Tagger Plugin."""

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
        responses = call_model(blocks, hf_bearer_token=hf_bearer_token, hf_model_path=self.model_path)
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

        if request is None:
            return Response(error=SteamshipError(message="Missing PluginRequest"))

        if request.data is None:
            return Response(error=SteamshipError(message="Missing ParseRequest"))

        if request.data.file is None:
            return Response(error=SteamshipError(message="Missing File"))

        start_time = time.time()
        self.tag_blocks(request.data.file.blocks, self.config.get('hf_api_bearer_token', ''))
        total_time = time.time() - start_time
        logging.info(f'Completed {len(request.data.file.blocks)} blocks in {total_time} seconds. ({float(len(request.data.file.blocks))/total_time} bps)')

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
