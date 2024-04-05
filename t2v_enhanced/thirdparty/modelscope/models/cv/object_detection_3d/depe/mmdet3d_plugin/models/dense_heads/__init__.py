# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly available at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads
"""
from .petrv2_dednhead import PETRv2DEDNHead

__all__ = ['PETRv2DEDNHead']
