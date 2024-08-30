from .iimage import IImage

import math
import numpy as np
import warnings

# ========= STATIC FUNCTIONS =============
def find_max_h(images):
    return max([x.size[1] for x in images])
def find_max_w(images):
    return max([x.size[0] for x in images])
def find_max_size(images):
    return find_max_w(images), find_max_h(images)


def stack(images, axis = 0):
    return IImage(np.concatenate([x.data for x in images], axis))
def tstack(images):
    w,h = find_max_size(images)
    images = [x.pad2wh(w,h) for x in images]
    return IImage(np.concatenate([x.data for x in images], 0))
def hstack(images):
    h = find_max_h(images)
    images = [x.pad2wh(h = h) for x in images]
    return IImage(np.concatenate([x.data for x in images], 2))
def vstack(images):
    w = find_max_w(images)
    images = [x.pad2wh(w = w) for x in images]
    return IImage(np.concatenate([x.data for x in images], 1))

def grid(images, nrows = None, ncols = None):
    combined = stack(images)
    if nrows is not None:
        ncols = math.ceil(combined.data.shape[0] / nrows)
    elif ncols is not None:
        nrows = math.ceil(combined.data.shape[0] / ncols)
    else:
        warnings.warn("No dimensions specified, creating a grid with 5 columns (default)")
        ncols = 5
        nrows = math.ceil(combined.data.shape[0] / ncols)
        
    pad = nrows * ncols - combined.data.shape[0]
    data = np.pad(combined.data, ((0,pad),(0,0),(0,0),(0,0)))
    rows = [np.concatenate(x,1,dtype=np.uint8) for x in np.array_split(data, nrows)]
    return IImage(np.concatenate(rows, 0, dtype = np.uint8)[None])