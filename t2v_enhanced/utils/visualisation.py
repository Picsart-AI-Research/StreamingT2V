from collections import defaultdict
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

def pil_concat_v(images):
    width = images[0].width
    height = sum([image.height for image in images])
    dst = Image.new('RGB', (width, height))
    h = 0
    for image_idx, image in enumerate(images):
        dst.paste(image, (0, h))
        h += image.height
    return dst

def pil_concat_h(images):
    width = sum([image.width for image in images])
    height = images[0].height
    dst = Image.new('RGB', (width, height))
    w = 0
    for image_idx, image in enumerate(images):
        dst.paste(image, (w, 0))
        w += image.width
    return dst

def add_label(image, text, fontsize=12):
    dst = Image.new('RGB', (image.width, image.height + fontsize*3))
    dst.paste(image, (0, 0))
    draw = ImageDraw.Draw(dst)
    font = ImageFont.truetype("../misc/fonts/OpenSans.ttf", fontsize)
    draw.text((fontsize, image.height + fontsize),text,(255,255,255),font=font)    
    return dst

def pil_concat(images, labels=None, col=8, fontsize=12):
    col = min(col, len(images))
    if labels is not None:
        labeled_images = [add_label(image, labels[image_idx], fontsize=fontsize) for image_idx, image in enumerate(images)]
    else:
        labeled_images = images
    labeled_images_rows = []
    for row_idx in range(int(np.ceil(len(labeled_images) / col))):
        labeled_images_rows.append(pil_concat_h(labeled_images[col*row_idx:col*(row_idx+1)]))
    return pil_concat_v(labeled_images_rows)


def draw_panoptic_segmentation(model, segmentation, segments_info):
    # get the used color map
    viridis = cm.get_cmap('viridis')
    norm = Normalize(vmin=segmentation.min().item(), vmax=segmentation.max().item())
    fig, ax = plt.subplots()
    ax.imshow(segmentation, cmap=viridis, norm=norm)
    instances_counter = defaultdict(int)
    handles = []
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(norm(segment_id))
        handles.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=handles)



rescale_ = lambda x: (x + 1.) / 2.

def pil_grid_display(x, mask=None, nrow=4, rescale=True):
    if rescale:
        x = rescale_(x)
    if mask is not None:
        mask = mask_to_3_channel(mask)
        x = torch.concat([mask, x])
    grid = make_grid(torch.clip(x, 0, 1), nrow=nrow)
    return ToPILImage()(grid)

def pil_display(x, rescale=True):
    if rescale:
        x = rescale_(x)
    image = torch.clip(rescale_(x), 0, 1)
    return ToPILImage()(image)

def mask_to_3_channel(mask):
    if mask.dim() == 3:
        mask_c_idx = 0
    elif mask.dim() == 4:
        mask_c_idx = 1
    else:
        raise Exception("mask should be a 3d or 4d tensor")
    
    if mask.shape[mask_c_idx] == 3:
        return mask
    elif mask.shape[mask_c_idx] == 1:
        sizes = [1] * mask.dim()
        sizes[mask_c_idx] = 3
        mask = mask.repeat(*sizes) 
    else:
        raise Exception("mask should have size 1 in channel dim")
    return mask


def get_first_k_token_head_att_maps(atts_normed, k, h, w, output_h=256, output_w=256, labels=None, max_scale=False):
    n_heads = atts_normed.shape[0]
    att_images = []
    for head_idx in range(n_heads):
        atts_head = atts_normed[head_idx, :, :k].reshape(h, w, k).movedim(2, 0)
        for token_idx in range(k):
            att_head_np = atts_head[token_idx].detach().cpu().numpy()
            if max_scale:
                att_head_np = att_head_np / att_head_np.max()
            att_image = Image.fromarray((att_head_np * 255).astype(np.uint8))
            att_image = att_image.resize((output_h, output_w), Image.Resampling.NEAREST)
            att_images.append(att_image)
    return pil_concat(att_images, col=k, labels=None)

def get_first_k_token_att_maps(atts_normed, k, h, w, output_h=256, output_w=256, labels=None, max_scale=False):
    att_images = []
    atts_head = atts_normed.mean(0)[:, :k].reshape(h, w, k).movedim(2, 0)
    for token_idx in range(k):
        att_head_np = atts_head[token_idx].detach().cpu().numpy()
        if max_scale:
            att_head_np = att_head_np / att_head_np.max()
        att_image = Image.fromarray((att_head_np * 255).astype(np.uint8))
        att_image = att_image.resize((output_h, output_w), Image.Resampling.NEAREST)
        att_images.append(att_image)
    return pil_concat(att_images, col=k, labels=None)

def draw_bbox(image, bbox):
    image = image.copy()
    left, top, right, bottom = bbox
    image_draw = ImageDraw.Draw(image)
    image_draw.rectangle(((left, top),(right, bottom)), outline='Red')
    return image