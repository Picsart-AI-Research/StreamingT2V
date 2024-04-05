# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# The implementation is based on YOLOX, available at https://github.com/Megvii-BaseDetection/YOLOX

import os
import sys


def get_exp_by_name(exp_name):
    exp = exp_name.replace('-',
                           '_')  # convert string like "yolox-s" to "yolox_s"
    if exp == 'streamyolo':
        from .default import StreamYoloExp as YoloXExp
    else:
        pass
    return YoloXExp()
