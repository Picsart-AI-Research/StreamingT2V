# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import GPT2Config
from transformers import GPT2Model as GPT2ModelTransform

from modelscope.metainfo import Models
from modelscope.models.builder import BACKBONES
from modelscope.utils.constant import Tasks


@BACKBONES.register_module(group_key=Tasks.backbone, module_name=Models.gpt2)
class GPT2Model(GPT2ModelTransform):

    def __init__(self, **kwargs):
        config = GPT2Config(**kwargs)
        super().__init__(config)
