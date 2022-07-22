# Zero-Shot Classifier Plugin 
### https://huggingface.co/facebook/bart-large-mnli

This plugin wraps a (zero-shot classification model)[https://huggingface.co/models?pipeline_tag=zero-shot-classification] available on hugging face.

When you instantiate this plugin, you must provide it `labels` that will be used to label the data, and a `tag_kind` that will be used to identify the resulting tags.

Set `multi_label=True` if classes can overlap.

For example, if you wished to classify sentences about cats versus those about dogs, you could use:

```
labels='cats,dogs'
tag_kind='my-animal-classification'
```
and this would result in tags of the form:
```python
Tag(kind='my-animal-classification',name='dog',value={'score': 0.9})
```

The plugin will classify the text in every `Block` of the `File` that it receives as input. When used in multi-class mode (`multi_label=False`) the most probable `Tag` will be applied. When used in multi-label mode (`multi_label=True`), `Tags` for every possible label will be added, no matter the returned confidence.

To use this plugin, you must also provide a `hf_api_bearer_token` which will be used to invoke the model.

## Parameters

| Parameter | Description | DType | Required | Default |
|-------------------|----------------------------------------------------|--------|--|--|
| hf_api_bearer_token | Your bearer token from the Hugging Face API. | string |Yes| d |
| hf_model_path | Your bearer token from the Hugging Face API. | string |Yes| - |
| labels | A comma-separated list of labels that will be applied. | string |Yes|  - |
| tag_kind | A value for 'kind' in the Tag objects that will be returned. | string |Yes|  - |
| multi_label | Whether classification labels can overlap. | string |Yes| False |
| use_gpu | Process HF requests on GPU (at greater speed and cost). | string |No| False |

## Developing

Development instructions are located in [DEVELOPING.md](DEVELOPING.md)

## Testing

Testing instructions are located in [TESTING.md](TESTING.md)

## Deploying

Deployment instructions are located in [DEPLOYING.md](DEPLOYING.md)