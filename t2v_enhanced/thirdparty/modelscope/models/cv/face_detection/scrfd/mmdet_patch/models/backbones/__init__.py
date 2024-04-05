# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
"""
The implementation here is modified based on insightface, originally MIT license and publicly available at
https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet/models/backbones
"""
from .master_net import MasterNet
from .mobilenet import MobileNetV1
from .resnet import ResNetV1e

__all__ = ['ResNetV1e', 'MobileNetV1', 'MasterNet']
