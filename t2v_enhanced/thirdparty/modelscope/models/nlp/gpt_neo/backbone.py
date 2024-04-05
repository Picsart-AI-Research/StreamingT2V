# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import GPTNeoConfig
from transformers import GPTNeoModel as GPTNeoModelTransform

from modelscope.metainfo import Models
from modelscope.models.builder import BACKBONES
from modelscope.utils.constant import Tasks


@BACKBONES.register_module(
    group_key=Tasks.backbone, module_name=Models.gpt_neo)
class GPTNeoModel(GPTNeoModelTransform):

    def __init__(self, **kwargs):
        config = GPTNeoConfig(**kwargs)
        super().__init__(config)
