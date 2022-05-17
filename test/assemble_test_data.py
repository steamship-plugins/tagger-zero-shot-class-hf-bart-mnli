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



def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'example_text.json'),'r') as example_text_file:
        example_texts = json.load(example_text_file)

    with open(os.path.join(folder, '..', 'test_data', 'example_results.json'),'r') as results_file:
        results = json.load(results_file)


    with open(os.path.join(folder, '..', 'test_data', 'dataset.jsonl'),'w') as dataset_file:
        for i, text in enumerate(example_texts):
            result = results[i]
            if isinstance(result, list) and len(result) > 0:
                sorted_results = sorted(result, key=lambda x: x['score'], reverse=True)

                label = sorted_results[0]['entity_group']
                dataset_file.write(json.dumps( dict(classificationAnnotation= dict(displayName=label), textContent=text)))
                dataset_file.write("\n")

main()


