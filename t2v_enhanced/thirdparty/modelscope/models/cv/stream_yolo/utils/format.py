# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
import math


def timestamp_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    time = '%02d:%02d:%06.3f' % (h, m, s)
    return time
