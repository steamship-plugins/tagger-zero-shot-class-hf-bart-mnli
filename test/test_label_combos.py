from steamship.utils.huggingface_helper import get_huggingface_results
import os
import json
from src.api import TaggerPlugin
from steamship import Block


def test_same_results():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'test-config.json'), 'r') as config_file:
        config = json.load(config_file)
    config['multi_label'] = False


    text = 'I really enjoy seeing large animals in the wild, especially cats and animals with tusks.'
    labels = ['dog','groundhog','tiger','elephant','lion', 'mouse' ]

    labelsets = [[0,1,2],[3,4,5], [2,4,5], [1,3,4], [0,2,3], [5,0,1]]

    results = []
    for labelset in labelsets:
        this_call_labels = [label for i, label in enumerate(labels) if i in labelset]
        config['labels'] = ','.join(this_call_labels)
        tagger = TaggerPlugin(config=config)
        block = Block(text=text)
        tagger.tag_blocks([block], config['hf_api_bearer_token'])
        this_call_results = { tag.name : tag.value['score'] for tag in block.tags}
        results.append(this_call_results)

    label_results = { label : [result[label] for result in results if label in result] for label in labels}
    print(label_results)
    for label in labels:
        assert len(set(label_results[label])) == 1
        print(f'{label}: {label_results[label][0]}')

