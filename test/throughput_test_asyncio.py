from helper.hf_model_helper import call_model_on_text
import time
import json
import os
import logging
import requests
from steamship import SteamshipError
from time import sleep
import aiohttp
import asyncio
from typing import List

async def get_pokemon(session, url):
    async with session.get(url) as resp:
        pokemon = await resp.json()
        return pokemon['name']




async def model_call(session, text: str, api_url, headers):
    data = json.dumps(text)
    async with session.post(api_url, headers=headers, data=data) as response:
        json_response = await response.json()
        logging.info(json_response)
        return json_response

async def model_calls(texts: List[str], api_url : str, headers):

    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts:
            tasks.append(asyncio.ensure_future(model_call(session, text, api_url, headers=headers)))

        results = await asyncio.gather(*tasks)
        return results

def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'example_text.json'),'r') as example_text_file:
        example_texts = json.load(example_text_file)

    with open(os.path.join(folder, '..', 'test_data', 'test-config.json'), 'r') as config_file:
        config = json.load(config_file)

    hf_bearer_token = config.get('hf_api_bearer_token', '')

    request_texts = example_texts[:1000]
    api_url = "https://api-inference.huggingface.co/models/dslim/bert-large-NER"
    headers = {"Authorization": f"Bearer {hf_bearer_token}"}


    start_time = time.time()
    #call_model_on_text(request_texts, hf_bearer_token=hf_bearer_token, hf_model_path="dslim/bert-large-NER", hf_compute_type='gpu')
    # for text in request_texts:
    #     model_call(text, api_url, headers)
    results = asyncio.run(model_calls(request_texts, api_url, headers))
    total_time = time.time() - start_time
    print(f'Completed {len(request_texts)} blocks in {total_time} seconds. ({float(len(request_texts))/total_time} bps)')

    with open(os.path.join(folder, '..', 'test_data', 'example_results.json'),'w') as results_file:
        json.dump(results, results_file)

main()


