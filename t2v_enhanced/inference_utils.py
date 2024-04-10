# import argparse
import sys
from pathlib import Path
from pytorch_lightning.cli import LightningCLI
from PIL import Image

# For streaming
import yaml
from copy import deepcopy
from typing import List, Optional
from jsonargparse.typing import restricted_string_type


# --------------------------------------
# ----------- For Streaming ------------
# --------------------------------------
class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--result_fol", type=Path,
                            help="Set the path to the result folder", default="results")
        parser.add_argument("--exp_name", type=str, help="Experiment name")
        parser.add_argument("--run_name", type=str,
                            help="Current run name")
        parser.add_argument("--prompts", type=Optional[List[str]])
        parser.add_argument("--scale_lr", type=bool,
                            help="Scale lr", default=False)
        CodeType = restricted_string_type(
            'CodeType', '(medium)|(high)|(highest)')
        parser.add_argument("--matmul_precision", type=CodeType)
        parser.add_argument("--ckpt", type=Path,)
        parser.add_argument("--n_predictions", type=int)
        return parser

def remove_value(dictionary, x):
    for key, value in list(dictionary.items()):
        if key == x:
            del dictionary[key]
        elif isinstance(value, dict):
            remove_value(value, x)
    return dictionary

def legacy_transformation(cfg: yaml):
    cfg = deepcopy(cfg)
    cfg["trainer"]["devices"] = "1"
    cfg["trainer"]['num_nodes'] = 1

    if not "class_path" in cfg["model"]["inference_params"]:
        cfg["model"]["inference_params"] = {
            "class_path": "t2v_enhanced.model.pl_module_params.InferenceParams", "init_args": cfg["model"]["inference_params"]}
    return cfg


# ---------------------------------------------
# ----------- For enhancement -----------
# ---------------------------------------------
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def resize_to_fit(image, size):
    W, H = size
    w, h = image.size
    if H / h > W / w:
        H_ = int(h * W / w)
        W_ = W
    else:
        W_ = int(w * H / h)
        H_ = H
    return image.resize((W_, H_))

def pad_to_fit(image, size):
    W, H = size
    w, h = image.size
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    return add_margin(image, pad_h, pad_w, pad_h, pad_w, (0, 0, 0))

def resize_and_keep(pil_img):
    myheight = 576
    hpercent = (myheight/float(pil_img.size[1]))
    wsize = int((float(pil_img.size[0])*float(hpercent)))
    pil_img = pil_img.resize((wsize, myheight))
    return pil_img

def center_crop(pil_img):
    width, height = pil_img.size
    new_width = 576
    new_height = 576

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    pil_img = pil_img.crop((left, top, right, bottom))
    return pil_img


def v2v_to_device(pipe_enhance, device):
    pipe_enhance.device = device

    pipe_enhance.model = pipe_enhance.model.to(device)
    pipe_enhance.model.device = device
    
    pipe_enhance.model.clip_encoder.model = pipe_enhance.model.clip_encoder.model.to(device)
    pipe_enhance.model.clip_encoder.device = device

    pipe_enhance.model.autoencoder = pipe_enhance.model.autoencoder.to(device)
    pipe_enhance.model.generator = pipe_enhance.model.generator.to(device)
    if device.startswith("cuda"):
        pipe_enhance.model.generator = pipe_enhance.model.generator.half()
    pipe_enhance.model.negative_y = pipe_enhance.model.negative_y.to(device)
    return pipe_enhance

def st2v_to_device(stream_model, device):
    stream_model = stream_model.to(device)
    stream_model.inference_pipeline.unet = stream_model.inference_pipeline.unet.to(device)
    stream_model.inference_pipeline.vae = stream_model.inference_pipeline.vae.to(device)
    stream_model.inference_pipeline = stream_model.inference_pipeline.to(device)
    return stream_model