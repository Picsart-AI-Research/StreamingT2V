import io
import math
import os
import PIL.Image
import numpy as np
import imageio.v3 as iio
import warnings


import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import binary_dilation, binary_erosion
import cv2

import re

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, Image, display


IMG_THUMBSIZE = None

def torch2np(x, vmin=-1, vmax=1):
    if x.ndim != 4:
        # raise Exception("Please only use (B,C,H,W) torch tensors!")
        warnings.warn(
            "Warning! Shape of the image was not provided in (B,C,H,W) format, the shape was inferred automatically!")
        if x.ndim == 3:
            x = x[None]
        if x.ndim == 2:
            x = x[None, None]
    x = x.detach().cpu().float()
    if x.dtype == torch.uint8:
        return x.numpy().astype(np.uint8)
    elif vmin is not None and vmax is not None:
        x = (255 * (x.clip(vmin, vmax) - vmin) / (vmax - vmin))
        x = x.permute(0, 2, 3, 1).to(torch.uint8)
        return x.numpy()
    else:
        raise NotImplementedError()


class IImage:
    '''
    Generic media storage. Can store both images and videos.
    Stores data as a numpy array by default.
    Can be viewed in a jupyter notebook.
    '''
    @staticmethod
    def open(path):

        iio_obj = iio.imopen(path, 'r')
        data = iio_obj.read()
        try:
            # .properties() does not work for images but for gif files
            if not iio_obj.properties().is_batch:
                data = data[None]
        except AttributeError as e:
            # this one works for gif files
            if not "duration" in iio_obj.metadata():
                data = data[None]
        if data.ndim == 3:
            data = data[..., None]
        image = IImage(data)
        image.link = os.path.abspath(path)
        return image

    @staticmethod
    def normalized(x, dims=[-1, -2]):
        x = (x - x.amin(dims, True)) / \
            (x.amax(dims, True) - x.amin(dims, True))
        return IImage(x, 0)

    def numpy(self): return self.data

    def torch(self, vmin=-1, vmax=1):
        if self.data.ndim == 3:
            data = self.data.transpose(2, 0, 1) / 255.
        else:
            data = self.data.transpose(0, 3, 1, 2) / 255.
        return vmin + torch.from_numpy(data).float().to(self.device) * (vmax - vmin)

    def cuda(self):
        self.device = 'cuda'
        return self

    def cpu(self):
        self.device = 'cpu'
        return self

    def pil(self):
        ans = []
        for x in self.data:
            if x.shape[-1] == 1:
                x = x[..., 0]

            ans.append(PIL.Image.fromarray(x))
        if len(ans) == 1:
            return ans[0]
        return ans

    def is_iimage(self):
        return True

    @property
    def shape(self): return self.data.shape
    @property
    def size(self): return (self.data.shape[-2], self.data.shape[-3])

    def setFps(self, fps):
        self.fps = fps
        self.generate_display()
        return self

    def __init__(self, x, vmin=-1, vmax=1, fps=None):
        if isinstance(x, PIL.Image.Image):
            self.data = np.array(x)
            if self.data.ndim == 2:
                self.data = self.data[..., None]  # (H,W,C)
            self.data = self.data[None]  # (B,H,W,C)
        elif isinstance(x, IImage):
            self.data = x.data.copy()  # Simple Copy
        elif isinstance(x, np.ndarray):
            self.data = x.copy().astype(np.uint8)
            if self.data.ndim == 2:
                self.data = self.data[None, ..., None]
            if self.data.ndim == 3:
                warnings.warn(
                    "Inferred dimensions for a 3D array as (H,W,C), but could've been (B,H,W)")
                self.data = self.data[None]
        elif isinstance(x, torch.Tensor):
            self.data = torch2np(x, vmin, vmax)
        self.display_str = None
        self.device = 'cpu'
        self.fps = fps if fps is not None else (
            1 if len(self.data) < 10 else 30)
        self.link = None

    def generate_display(self):
        if IMG_THUMBSIZE is not None:
            if self.size[1] < self.size[0]:
                thumb = self.resize(
                    (self.size[1]*IMG_THUMBSIZE//self.size[0], IMG_THUMBSIZE))
            else:
                thumb = self.resize(
                    (IMG_THUMBSIZE, self.size[0]*IMG_THUMBSIZE//self.size[1]))
        else:
            thumb = self
        if self.is_video():
            self.anim = Animation(thumb.data, fps=self.fps)
            self.anim.render()
            self.display_str = self.anim.anim_str
        else:
            b = io.BytesIO()
            data = thumb.data[0]
            if data.shape[-1] == 1:
                data = data[..., 0]
            PIL.Image.fromarray(data).save(b, "PNG")
            self.display_str = b.getvalue()
        return self.display_str

    def resize(self, size, *args, **kwargs):
        if size is None:
            return self
        use_small_edge_when_int = kwargs.pop('use_small_edge_when_int', False)

        # Backward compatibility
        resample = kwargs.pop('filter', PIL.Image.BICUBIC)
        resample = kwargs.pop('resample', resample)

        if isinstance(size, int):
            if use_small_edge_when_int:
                h, w = self.data.shape[1:3]
                aspect_ratio = h / w
                size = (max(size, int(size * aspect_ratio)),
                        max(size, int(size / aspect_ratio)))
            else:
                h, w = self.data.shape[1:3]
                aspect_ratio = h / w
                size = (min(size, int(size * aspect_ratio)),
                        min(size, int(size / aspect_ratio)))

        if self.size == size[::-1]:
            return self
        return stack([IImage(x.pil().resize(size[::-1], *args, resample=resample, **kwargs)) for x in self])

    def pad(self, padding, *args, **kwargs):
        return IImage(TF.pad(self.torch(0), padding=padding, *args, **kwargs), 0)

    def padx(self, multiplier, *args, **kwargs):
        size = np.array(self.size)
        padding = np.concatenate(
            [[0, 0], np.ceil(size / multiplier).astype(int) * multiplier - size])
        return self.pad(list(padding), *args, **kwargs)

    def pad2wh(self, w=0, h=0, **kwargs):
        cw, ch = self.size
        return self.pad([0, 0, max(0, w - cw), max(0, h-ch)], **kwargs)

    def pad2square(self, *args, **kwargs):
        if self.size[0] > self.size[1]:
            dx = self.size[0] - self.size[1]
            return self.pad([0, dx//2, 0, dx-dx//2], *args, **kwargs)
        elif self.size[0] < self.size[1]:
            dx = self.size[1] - self.size[0]
            return self.pad([dx//2, 0, dx-dx//2, 0], *args, **kwargs)
        return self

    def crop2square(self, *args, **kwargs):
        if self.size[0] > self.size[1]:
            dx = self.size[0] - self.size[1]
            return self.crop([dx//2, 0, self.size[1], self.size[1]], *args, **kwargs)
        elif self.size[0] < self.size[1]:
            dx = self.size[1] - self.size[0]
            return self.crop([0, dx//2, self.size[0], self.size[0]], *args, **kwargs)
        return self

    def alpha(self):
        return IImage(self.data[..., -1, None], fps=self.fps)

    def rgb(self):
        return IImage(self.pil().convert('RGB'), fps=self.fps)

    def png(self):
        return IImage(np.concatenate([self.data, 255 * np.ones_like(self.data)[..., :1]], -1))

    def grid(self, nrows=None, ncols=None):
        if nrows is not None:
            ncols = math.ceil(self.data.shape[0] / nrows)
        elif ncols is not None:
            nrows = math.ceil(self.data.shape[0] / ncols)
        else:
            warnings.warn(
                "No dimensions specified, creating a grid with 5 columns (default)")
            ncols = 5
            nrows = math.ceil(self.data.shape[0] / ncols)

        pad = nrows * ncols - self.data.shape[0]
        data = np.pad(self.data, ((0, pad), (0, 0), (0, 0), (0, 0)))
        rows = [np.concatenate(x, 1, dtype=np.uint8)
                for x in np.array_split(data, nrows)]
        return IImage(np.concatenate(rows, 0, dtype=np.uint8)[None])

    def hstack(self):
        return IImage(np.concatenate(self.data, 1, dtype=np.uint8)[None])

    def vstack(self):
        return IImage(np.concatenate(self.data, 0, dtype=np.uint8)[None])

    def vsplit(self, number_of_splits):
        return IImage(np.concatenate(np.split(self.data, number_of_splits, 1)))

    def hsplit(self, number_of_splits):
        return IImage(np.concatenate(np.split(self.data, number_of_splits, 2)))

    def heatmap(self, resize=None, cmap=cv2.COLORMAP_JET):
        data = np.stack([cv2.cvtColor(cv2.applyColorMap(
            x, cmap), cv2.COLOR_BGR2RGB) for x in self.data])
        return IImage(data).resize(resize, use_small_edge_when_int=True)

    def display(self):
        try:
            display(self)
        except:
            print("No display")
        return self

    def dilate(self, iterations=1, *args, **kwargs):
        if iterations == 0:
            return IImage(self.data)
        return IImage((binary_dilation(self.data, iterations=iterations, *args, *kwargs)*255.).astype(np.uint8))

    def erode(self, iterations=1, *args, **kwargs):
        return IImage((binary_erosion(self.data, iterations=iterations, *args, *kwargs)*255.).astype(np.uint8))

    def hull(self):
        convex_hulls = []
        for frame in self.data:
            contours, hierarchy = cv2.findContours(
                frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [x.astype(np.int32) for x in contours]
            mask_contours = [cv2.convexHull(np.concatenate(contours))]
            canvas = np.zeros(self.data[0].shape, np.uint8)
            convex_hull = cv2.drawContours(
                canvas, mask_contours, -1, (255, 0, 0), -1)
            convex_hulls.append(convex_hull)
        return IImage(np.array(convex_hulls))

    def is_video(self):
        return self.data.shape[0] > 1

    def __getitem__(self, idx):
        return IImage(self.data[None, idx], fps=self.fps)
        # if self.is_video(): return IImage(self.data[idx], fps = self.fps)
        # return self

    def _repr_png_(self):
        if self.is_video():
            return None
        if self.display_str is None:
            self.generate_display()
        return self.display_str

    def _repr_html_(self):
        if not self.is_video():
            return None
        if self.display_str is None:
            self.generate_display()
        return self.display_str

    def save(self, path):
        _, ext = os.path.splitext(path)
        if self.is_video():
            # if ext in ['.jpg', '.png']:
            if self.display_str is None:
                self.generate_display()
            if ext == ".apng":
                self.anim.anim_obj.save(path, writer="pillow")
            else:
                self.anim.anim_obj.save(path)
        else:
            data = self.data if self.data.ndim == 3 else self.data[0]
            if data.shape[-1] == 1:
                data = data[:, :, 0]
            PIL.Image.fromarray(data).save(path)
        return self

    def write(self, text, center=(0, 25), font_scale=0.8, color=(255, 255, 255), thickness=2):
        if not isinstance(text, list):
            text = [text for _ in self.data]
        data = np.stack([cv2.putText(x.copy(), t, center, cv2.FONT_HERSHEY_COMPLEX,
                        font_scale, color, thickness) for x, t in zip(self.data, text)])
        return IImage(data)

    def append_text(self, text, padding, font_scale=0.8, color=(255, 255, 255), thickness=2, scale_factor=0.9, center=(0, 0), fill=0):

        assert np.count_nonzero(padding) == 1
        axis_padding = np.nonzero(padding)[0][0]
        scale_padding = padding[axis_padding]

        y_0 = 0
        x_0 = 0
        if axis_padding == 0:
            width = scale_padding
            y_max = self.shape[1]
        elif axis_padding == 1:
            width = self.shape[2]
            y_max = scale_padding
        elif axis_padding == 2:
            x_0 = self.shape[2]
            width = scale_padding
            y_max = self.shape[1]
        elif axis_padding == 3:
            width = self.shape[2]
            y_0 = self.shape[1]
            y_max = self.shape[1]+scale_padding

        width -= center[0]
        x_0 += center[0]
        y_0 += center[1]

        self = self.pad(padding, fill=fill)

        def wrap_text(text, width, _font_scale):
            allowed_seperator = ' |-|_|/|\n'
            words = re.split(allowed_seperator, text)
            # words = text.split()
            lines = []
            current_line = words[0]
            sep_list = []
            start_idx = 0
            for start_word in words[:-1]:
                pos = text.find(start_word, start_idx)
                pos += len(start_word)
                sep_list.append(text[pos])
                start_idx = pos+1

            for word, separator in zip(words[1:], sep_list):
                if cv2.getTextSize(current_line + separator + word, cv2.FONT_HERSHEY_COMPLEX, _font_scale, thickness)[0][0] <= width:
                    current_line += separator + word
                else:
                    if cv2.getTextSize(current_line, cv2.FONT_HERSHEY_COMPLEX, _font_scale, thickness)[0][0] <= width:
                        lines.append(current_line)
                        current_line = word
                    else:
                        return []

            if cv2.getTextSize(current_line, cv2.FONT_HERSHEY_COMPLEX, _font_scale, thickness)[0][0] <= width:
                lines.append(current_line)
            else:
                return []
            return lines

        def wrap_text_and_scale(text, width, _font_scale, y_0, y_max):
            height = y_max+1
            while height > y_max:
                text_lines = wrap_text(text, width, _font_scale)
                if len(text) > 0 and len(text_lines) == 0:

                    height = y_max+1
                else:
                    line_height = cv2.getTextSize(
                        text_lines[0], cv2.FONT_HERSHEY_COMPLEX, _font_scale, thickness)[0][1]
                    height = line_height * len(text_lines) + y_0

                # scale font if out of frame
                if height > y_max:
                    _font_scale = _font_scale * scale_factor

            return text_lines, line_height, _font_scale

        result = []
        if not isinstance(text, list):
            text = [text for _ in self.data]
        else:
            assert len(text) == len(self.data)

        for x, t in zip(self.data, text):
            x = x.copy()
            text_lines, line_height, _font_scale = wrap_text_and_scale(
                t, width, font_scale, y_0, y_max)
            y = line_height
            for line in text_lines:
                x = cv2.putText(
                    x, line, (x_0, y_0+y), cv2.FONT_HERSHEY_COMPLEX, _font_scale, color, thickness)
                y += line_height
            result.append(x)
        data = np.stack(result)

        return IImage(data)

    # ========== OPERATORS =============

    def __or__(self, other):
        # TODO: fix for variable sizes
        return IImage(np.concatenate([self.data, other.data], 2))

    def __truediv__(self, other):
        # TODO: fix for variable sizes
        return IImage(np.concatenate([self.data, other.data], 1))

    def __and__(self, other):
        return IImage(np.concatenate([self.data, other.data], 0))

    def __add__(self, other):
        return IImage(0.5 * self.data + 0.5 * other.data)

    def __mul__(self, other):
        if isinstance(other, IImage):
            return IImage(self.data / 255. * other.data)
        return IImage(self.data * other / 255.)

    def __xor__(self, other):
        return IImage(0.5 * self.data + 0.5 * other.data + 0.5 * self.data * (other.data.sum(-1, keepdims=True) == 0))

    def __invert__(self):
        return IImage(255 - self.data)
    __rmul__ = __mul__

    def bbox(self):
        return [cv2.boundingRect(x) for x in self.data]

    def fill_bbox(self, bbox_list, fill=255):
        data = self.data.copy()
        for bbox in bbox_list:
            x, y, w, h = bbox
            data[:, y:y+h, x:x+w, :] = fill
        return IImage(data)

    def crop(self, bbox):
        assert len(bbox) in [2, 4]
        if len(bbox) == 2:
            x, y = 0, 0
            w, h = bbox
        elif len(bbox) == 4:
            x, y, w, h = bbox
        return IImage(self.data[:, y:y+h, x:x+w, :])

def stack(images, axis = 0):
    return IImage(np.concatenate([x.data for x in images], axis))

class Animation:
    JS = 0
    HTML = 1
    ANIMATION_MODE = HTML
    def __init__(self, frames, fps = 30):
        """_summary_

        Args:
            frames (np.ndarray): _description_
        """        
        self.frames = frames
        self.fps = fps
        self.anim_obj = None
        self.anim_str = None
    def render(self):
        size = (self.frames.shape[2],self.frames.shape[1])
        self.fig = plt.figure(figsize = size, dpi = 1)
        plt.axis('off')
        img = plt.imshow(self.frames[0], cmap = 'gray')
        self.fig.subplots_adjust(0,0,1,1)
        self.anim_obj = animation.FuncAnimation(
            self.fig, 
            lambda i: img.set_data(self.frames[i,:,:,:]),
            frames=self.frames.shape[0], 
            interval = 1000 / self.fps
        )
        plt.close()
        if Animation.ANIMATION_MODE == Animation.HTML:
            self.anim_str = self.anim_obj.to_html5_video(embed_limit=1000.0)
        elif Animation.ANIMATION_MODE == Animation.JS:
            self.anim_str = self.anim_obj.to_jshtml()
        return self.anim_obj
    def _repr_html_(self):
        if self.anim_obj is None: self.render()
        return self.anim_str