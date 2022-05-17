from helper.hf_model_helper import call_model_on_text
import time
import json
import os
import logging
import requests
from steamship import SteamshipError
from time import sleep

def model_call(text: str, api_url, headers):
    data = json.dumps(text)
    valid_response = None
    while valid_response is None:
        response = requests.request("POST", api_url, headers=headers, data=data)
        json_response = json.loads(response.content.decode("utf-8"))
        logging.info(json_response)
        if 'error' in json_response:
            if not ('is currently loading' in json_response['error']):
                raise SteamshipError(message='Unable to query Hugging Face model',
                                     internalMessage=f'HF returned error: {json_response["error"]}')
            else:
                sleep(5)
        else:
            valid_response = json_response
    return valid_response

def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'example_text.json'),'r') as example_text_file:
        example_texts = json.load(example_text_file)

    with open(os.path.join(folder, '..', 'test_data', 'test-config.json'), 'r') as config_file:
        config = json.load(config_file)

    hf_bearer_token = config.get('hf_api_bearer_token', '')

    request_texts = example_texts[:100]
    api_url = "https://api-inference.huggingface.co/models/dslim/bert-large-NER"
    headers = {"Authorization": f"Bearer {hf_bearer_token}"}


    start_time = time.time()
    #call_model_on_text(request_texts, hf_bearer_token=hf_bearer_token, hf_model_path="dslim/bert-large-NER", hf_compute_type='gpu')
    for text in request_texts:
        model_call(text, api_url, headers)
    total_time = time.time() - start_time
    print(f'Completed {len(request_texts)} blocks in {total_time} seconds. ({float(len(request_texts))/total_time} bps)')

main()


