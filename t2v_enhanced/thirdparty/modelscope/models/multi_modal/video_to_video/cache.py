# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
import torch
import logging
from typing import List, Dict, Any
import pickle
import pathlib
import tempfile
import os


class TempCacheRAM:
    def __init__(self):
        self.data: Dict[Any, torch.Tensor] = {}

    def release(self):
        size = sum([t.element_size() * t.nelement() for t in self.data.values()])
        logging.info(f"Deleting temporal cache RAM count {len(self.data)} with size {size // 1024 // 1024} MB")
        self.data = {}

    def __del__(self):
        self.release()

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]


class TempCachePermanent:
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


def pick_cache_method():
    if "USE_RAM_CACHE" in os.environ and os.environ["USE_RAM_CACHE"]:
        logging.info("Using RAM cache")
        return TempCacheRAM()
    else:
        logging.info("Using Permanent memory for cache")
        return TempCachePermanent()
