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
import json
import asyncio
import aiohttp

async def _model_call(session, text: str, api_url, headers) -> list:
    json_input = dict(inputs=text, wait_for_model=True)
    data = json.dumps(json_input)

    """
    Hugging Face returns an error that says that the model is currently loading
    if it believes you have 'too many' requests simultaneously, so the logic retries in this case, but fails on
    other errors.
    """
    while True:
        async with session.post(api_url, headers=headers, data=data) as response:
            if response.status == 200 and response.content_type == 'application/json':
                    json_response = await response.json()
                    logging.info(json_response)
                    return json_response
            else:
                text_response = await response.text()
                logging.info(text_response)
                if "is currently loading" not in text_response:
                    raise SteamshipError(
                        message="Unable to query Hugging Face model",
                        internal_message=f'HF returned error: {text_response}',
                    )
                else:
                    await asyncio.sleep(1)


async def model_calls(texts: List[str], api_url : str, headers):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts:
            tasks.append(asyncio.ensure_future(_model_call(session, text, api_url, headers=headers)))

        results = await asyncio.gather(*tasks)
        return results

def get_results(blocks: List[Block], hf_model_path: str, hf_bearer_token : str):
    api_url = f"https://api-inference.huggingface.co/models/{hf_model_path}"
    headers = {"Authorization": f"Bearer {hf_bearer_token}"}
    start_time = time.time()
    results = asyncio.run(model_calls([block.text for block in blocks], api_url, headers))
    total_time = time.time() - start_time
    logging.info(
        f'Completed {len(blocks)} blocks in {total_time} seconds. ({float(len(blocks)) / total_time} bps)')
    return results

class TaggerPlugin(Tagger, App):
    """Example Steamship Tagger Plugin."""

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
        responses = get_results(blocks, hf_bearer_token=hf_bearer_token, hf_model_path=self.model_path)
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


handler = create_handler(TaggerPlugin)
