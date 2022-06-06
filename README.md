# Zero-Shot Classifier Plugin 
### https://huggingface.co/facebook/bart-large-mnli

This plugin wraps the `facebook/bart-large-mnli` zero-shot classification model available on hugging face.

When you instantiate this plugin, you must provide it `labels` that will be used to label the data, and a `tag_kind` that will be used in resulting tags.

For example, if you wished to classify sentences about cats versus those about dogs, you could use:

```
labels='cats,dogs'
tag_kind='my-animal-classification'
```
and this would result in tags of the form:
```python
Tag(kind='my-animal-classification',name='dog',value={'score': 0.9})
```

The plugin will classify the text in every `Block` of the `File` that it receives as input, and will provide `Tags` for every possible label, no matter the returned confidence.

To use this plugin, you must also provide a `hf_api_bearer_token` which will be used to invoke the model.

