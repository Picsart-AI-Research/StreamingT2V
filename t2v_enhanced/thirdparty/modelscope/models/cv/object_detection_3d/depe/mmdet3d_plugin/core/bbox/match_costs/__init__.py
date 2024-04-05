# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly available at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/core/bbox/match_costs
"""
from .match_cost import BBox3DL1Cost

__all__ = ['BBox3DL1Cost']
