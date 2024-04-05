# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# The implementation is adopted from VitAdapter,
# made publicly available under the Apache License at https://github.com/czczup/ViT-Adapter.git
from .backbone import BASEBEiT, BEiTAdapter
from .decode_heads import Mask2FormerHeadFromMMSeg
from .segmentors import EncoderDecoderMask2Former
