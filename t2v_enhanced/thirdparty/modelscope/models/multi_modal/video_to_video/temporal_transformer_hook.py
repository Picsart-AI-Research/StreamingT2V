# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
from enum import Flag, auto
from .modules.unet_v2v import MemoryEfficientCrossAttention
from typing import Callable, Sequence, Dict, Hashable, Any
import torch
import shelve
import tempfile
import logging
import pathlib
import os
import pickle


class TemporalAttentionHookMode(Flag):
    NONE = 0
    READ = auto()
    WRITE = auto()



class TempCache:

    def __init__(self):
        self.files: Dict[Any, pathlib.Path] = {}
        self._dir = tempfile.mkdtemp()

    def create_file(self, key):
        self.files[key] = pathlib.Path(tempfile.mktemp(dir=self._dir))

    def release(self):
        files_to_delete = filter(lambda x: x.exists(), self.files.values())
        files_to_delete = list(files_to_delete)

        size = sum([file.stat().st_size for file in files_to_delete])

        logging.info(f"Deleting temporal cache file count {len(files_to_delete)} with size {size // 1024 // 1024} MB")

        for file in files_to_delete:
            file.unlink()

    def __del__(self):
        self.release()

    def __setitem__(self, key, value):
        if key not in self.files:
            self.create_file(key)
        self.files[key].write_bytes(pickle.dumps(value))

    def __getitem__(self, key):
        return pickle.loads(self.files[key].read_bytes())


class TemporalAttentionHookManager:
    def __init__(self):
        self.cache = TempCache()
        self.layer_ids = []
        self.step_layer_id = 0
        self.hook_mode = TemporalAttentionHookMode.NONE

    def set_hook_mode(self, mode: TemporalAttentionHookMode):
        self.hook_mode = mode

    def reset_step_counter(self):
        self.step_layer_id = 0

    def temp_att_read_write_hook(self, name: str) -> Callable:
        self.layer_ids.append(name)

        def t_block_forward(module, x_val, x_att):
            module.only_self_att = False
            output = module.forward(x_val, context=x_att)
            module.only_self_att = True
            return output

        @torch.no_grad()
        def hook(module: MemoryEfficientCrossAttention,
                 input_tensor: torch.Tensor,
                 output):
            query = (name, self.step_layer_id)
            tensor_to_attend = input_tensor[0]
            if TemporalAttentionHookMode.READ in self.hook_mode:
                tensor_to_attend = self.cache[query].to(torch.float32).cuda()
            if TemporalAttentionHookMode.WRITE in self.hook_mode:
                self.cache[query] = input_tensor[0].to(torch.float16).detach().clone().cpu()

            out = t_block_forward(module,
                                  input_tensor[0],
                                  tensor_to_attend)
            output[:] = out[:]

            self.step_layer_id += 1
            return output
        return hook

    def clear_state(self):
        self.cache.release()
        self.hook_mode = TemporalAttentionHookMode.NONE
