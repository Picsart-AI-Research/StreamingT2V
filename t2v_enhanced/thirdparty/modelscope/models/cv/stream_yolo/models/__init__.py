# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# The implementation is based on YOLOX, available at https://github.com/Megvii-BaseDetection/YOLOX

from .darknet import CSPDarknet, Darknet
from .dfp_pafpn import DFPPAFPN
from .streamyolo import StreamYOLO
from .tal_head import TALHead
