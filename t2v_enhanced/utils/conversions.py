from pathlib import Path
import PIL
from PIL import Image
import numpy as np
from dataclasses import dataclass

# TODO add register new converter so that it is accessible via converters.to_x

def ensure_class(func, params):
    def func_wrapper(function):
        def wrapper(self=None, *args, **kwargs):
            for key in kwargs:
                if key in params:
                    kwargs[key] = func(kwargs[key])
            if self is not None:
                return function(self, *args, **kwargs)
            else:
                return function(*args, **kwargs)

        return wrapper

    return func_wrapper


def as_PIL(img):
    if not isinstance(img, PIL.Image.Image):
        if isinstance(img, Path):
            img = img.as_posix()
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        else:
            raise NotImplementedError
    return img


def to_ndarray(input):
    if not isinstance(input, np.ndarray):
        input = np.array(input)
    return input


def to_Path(input):
    if not isinstance(input, Path):
        input = Path(input)
    return input
