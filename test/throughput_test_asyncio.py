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


#Touch the model with a one-character test string until it is live
def ensure_model(api_url, headers):
    got_response = False
    start_time = time.time()
    while not got_response:
        response = requests.request("POST", api_url, headers=headers, data="t")
        json_response = json.loads(response.content.decode("utf-8"))
        logging.info(json_response)
        if 'error' in json_response:
            if not ('is currently loading' in json_response['error']):
                raise SteamshipError(message='Unable to query Hugging Face model',
                                     internalMessage=f'HF returned error: {json_response["error"]}')
            else:
                #sleep(1)
                pass
        else:
            got_response = True
    warmup_time = time.time() - start_time
    logging.info(f'Warmup time for model {warmup_time} seconds')


async def model_call(session, text: str, api_url, headers):
    input = dict(inputs=text, wait_for_model=True)
    data = json.dumps(input)
    valid_response = None
    while valid_response is None:
        async with session.post(api_url, headers=headers, data=data) as response:
            json_response = await response.json()
            logging.info(json_response)
            if 'error' in json_response:
                if not ('is currently loading' in json_response['error']):
                    raise SteamshipError(message='Unable to query Hugging Face model',
                                         internalMessage=f'HF returned error: {json_response["error"]}')
                else:
                    await asyncio.sleep(1)
            else:
                valid_response = json_response
    return valid_response

async def model_calls(texts: List[str], api_url : str, headers):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts:
            tasks.append(asyncio.ensure_future(model_call(session, text, api_url, headers=headers)))

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

    request_texts = example_texts[:100]

    results = get_results(request_texts, 'dslim/bert-large-NER', hf_bearer_token)

    with open(os.path.join(folder, '..', 'test_data', 'example_results.json'),'w') as results_file:
        json.dump(results, results_file)

main()


