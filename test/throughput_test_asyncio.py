from helper.hf_model_helper import call_model_on_text
import time
import json
import os
import logging
import requests
from steamship import SteamshipError
import aiohttp
import asyncio
from typing import List





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

def get_results(texts: List[str], model: str, hf_bearer_token : str):
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_bearer_token}"}
    #ensure_model(api_url, headers)
    start_time = time.time()
    results = asyncio.run(model_calls(texts, api_url, headers))
    total_time = time.time() - start_time
    logging.info(
        f'Completed {len(texts)} blocks in {total_time} seconds. ({float(len(texts)) / total_time} bps)')
    return results


def main():
    logging.basicConfig(level=logging.INFO)
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'example_text.json'),'r') as example_text_file:
        example_texts = json.load(example_text_file)

    with open(os.path.join(folder, '..', 'test_data', 'test-config.json'), 'r') as config_file:
        config = json.load(config_file)

    hf_bearer_token = config.get('hf_api_bearer_token', '')

    request_texts = example_texts[:1000]

    results = get_results(request_texts, 'dslim/bert-large-NER', hf_bearer_token)

    with open(os.path.join(folder, '..', 'test_data', 'example_results.json'),'w') as results_file:
        json.dump(results, results_file)

main()


