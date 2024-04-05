# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, List, Tuple, Union

Image = Union[str, 'Image.Image', 'numpy.ndarray']
Text = str
Audio = Union[str, bytes, 'np.ndarray']
Video = Union[str, 'np.ndarray', 'cv2.VideoCapture']

Tensor = Union['torch.Tensor', 'tf.Tensor']
