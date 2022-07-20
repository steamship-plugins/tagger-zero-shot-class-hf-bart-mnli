"""Example Steamship Tagger Plugin.

In Steamship, **Taggers** are responsible emitting tags that describe the **Steamship Block Format**.
"""
import logging
from itertools import zip_longest
from typing import List, Type

from steamship import Block, SteamshipError, Tag
from steamship.app import App, Response, create_handler
from steamship.plugin.config import Config
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import BlockAndTagPluginOutput
from steamship.plugin.service import PluginRequest
from steamship.plugin.tagger import Tagger
from steamship.utils.huggingface_helper import get_huggingface_results

TIMEOUT_SECONDS = 60
BATCH_SIZE: int = 10


class ZeroShotTaggerPlugin(Tagger, App):
    """Example Steamship Tagger Plugin."""

    class TaggerPluginConfig(Config):
        hf_api_bearer_token: str
        labels: str
        tag_kind: str
        multi_label: bool
        use_gpu: bool
        hf_model_path: str = "facebook/bart-large-mnli"

    def config_cls(self) -> Type[Config]:
        return self.TaggerPluginConfig

    def _make_tags_from_response(self, response: dict) -> List[Tag.CreateRequest]:
        return [
            Tag.CreateRequest(
                kind=self.config.tag_kind,
                name=label,
                value={"score": score},
            )
            for label, score in zip(response["labels"], response["scores"])
        ]

    @staticmethod
    def _batch_labels(labels: List[str]):
        return [labels[i : i + BATCH_SIZE] for i in range(0, len(labels), BATCH_SIZE)]

    def _tag_blocks(self, blocks: List[Block], hf_bearer_token: str):
        # Maximum of 10 labels per call to the model, but in multiclass, we can just make multiple calls to the model
        # with the same results.
        labels = self.config.labels.split(",")
        if len(labels) > BATCH_SIZE and not self.config.multi_label:
            raise SteamshipError(
                f"This plugin supports a maximum of {BATCH_SIZE} labels in multi-class classification."
            )

        for label_batch in self._batch_labels(labels):
            additional_parameters = {
                "candidate_labels": label_batch,
                "multi_label": self.config.multi_label,
            }
            responses = get_huggingface_results(
                blocks,
                hf_bearer_token=hf_bearer_token,
                hf_model_path=self.config.hf_model_path,
                additional_params=additional_parameters,
                timeout_seconds=TIMEOUT_SECONDS,
                use_gpu=self.config.use_gpu,
            )
            for block, response in zip_longest(blocks, responses):
                block.tags.extend(self._make_tags_from_response(response))

    def run(
        self, request: PluginRequest[BlockAndTagPluginInput]
    ) -> Response[BlockAndTagPluginOutput]:
        """Every plugin implements a `run` function.

        This plugin applies sentiment analysis via a pre-trained HF model
        to the text of each Block in a file.
        """
        logging.info("Invoking tagger-zero-shot-class-hf-bart-mnli")
        if request is None:
            return Response(error=SteamshipError(message="Missing PluginRequest"))

        if request.data is None:
            return Response(error=SteamshipError(message="Missing ParseRequest"))

        if request.data.file is None:
            return Response(error=SteamshipError(message="Missing File"))

        self._tag_blocks(request.data.file.blocks, self.config.hf_api_bearer_token)

        return Response(data=BlockAndTagPluginOutput(file=request.data.file))


handler = create_handler(ZeroShotTaggerPlugin)
