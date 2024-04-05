from tqdm import tqdm
from einops import repeat
from diffusers import DiffusionPipeline
from decord import VideoReader, cpu
import torchvision
import torch
import numpy as np
import decord
import albumentations as album
import math
import random
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union
from PIL import Image
import json
Image.MAX_IMAGE_PIXELS = None

decord.bridge.set_bridge("torch")

class Annotations():

    def __init__(self,
                 annotation_cfg: Dict) -> None:
        self.annotation_cfg = annotation_cfg

    # TODO find all special characters

    @staticmethod
    def process_string(string):
        for special_char in [".", ",", ":"]:
            result = ""
            i = 0
            while i < len(string):
                if string[i] == special_char:
                    if i > 0 and i < len(string) - 1 and string[i-1].isalpha() and string[i+1].isalpha():
                        result += special_char+" "
                    else:
                        result += special_char
                else:
                    result += string[i]
                i += 1
            string = result
        string = result
        return result

    @staticmethod
    def clean_prompt(prompt):
        prompt = " ".join(prompt.split())
        prompt = prompt.replace(" , ", ", ")
        prompt = prompt.replace(" . ", ". ")
        prompt = prompt.replace(" : ", ": ")
        prompt = Annotations.process_string(prompt)
        return prompt
        # return " ".join(prompt.split())

