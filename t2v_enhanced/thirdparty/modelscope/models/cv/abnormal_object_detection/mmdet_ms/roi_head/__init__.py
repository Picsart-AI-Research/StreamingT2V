# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
from .mask_scoring_roi_head import MaskScoringNRoIHead
from .roi_extractors import SingleRoINExtractor

__all__ = ['MaskScoringNRoIHead', 'SingleRoINExtractor']
