# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# The implementation is adopted from VitAdapter,
# made publicly available under the Apache License at https://github.com/czczup/ViT-Adapter.git
from .models import backbone, decode_heads, segmentors
from .utils import (ResizeToMultiple, add_prefix, build_pixel_sampler,
                    seg_resize)
