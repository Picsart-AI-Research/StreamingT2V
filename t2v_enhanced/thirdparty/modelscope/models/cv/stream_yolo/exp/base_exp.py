# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# The implementation is based on YOLOX, available at https://github.com/Megvii-BaseDetection/YOLOX

from abc import ABCMeta, abstractmethod

from torch.nn import Module


class BaseExp(metaclass=ABCMeta):

    @abstractmethod
    def get_model(self) -> Module:
        pass
