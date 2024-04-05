# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.

from .coco_register import register_all_coco
from .voc_register import register_all_voc


def register_data(data_type='pascal_voc', data_dir=None):

    if data_type == 'pascal_voc':
        if data_dir:
            register_all_voc(data_dir)
        else:
            register_all_voc()
    elif data_type == 'coco':
        if data_dir:
            register_all_coco(data_dir)
        else:
            register_all_coco()
    else:
        raise NotImplementedError(
            'no {} dataset was registered'.format(data_type))
