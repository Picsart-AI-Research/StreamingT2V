import importlib
from functools import partialmethod
from pathlib import Path
from torchvision.datasets.utils import download_url
import gdown
from utils.aux import ensure_annotation_class


def get_class(cls_path: str, *args, **kwargs):
    module_name = ".".join(cls_path.split(".")[:-1])
    module = importlib.import_module(module_name)

    class_ = getattr(module, cls_path.split(".")[-1])
    class_.__init__ = partialmethod(class_.__init__, *args, **kwargs)
    return class_


@ensure_annotation_class
def download_ckpt(local_path: Path, global_path: str) -> str:

    if local_path.exists():
        return local_path.as_posix()
    else:
        if not local_path.parent.exists():
            local_path.parent.mkdir(parents=True)

    if "drive.google.com" in global_path and "file" in global_path:
        url = global_path
        dest = local_path.as_posix()
        gdown.download(url=url, output=dest, fuzzy=True)

    elif "drive.google.com" in global_path and "folder" in global_path:
        url = global_path
        dest = local_path.parent.as_posix()
        gdown.download_folder(url=url, output=dest)

    elif local_path.suffix == ".safetensors" or "." not in local_path.as_posix():
        ckpt_url = f"https://huggingface.co/{global_path}"
        try:
            download_url(ckpt_url, local_path.parent.as_posix(),
                         local_path.name)
        except Exception as e:
            print(
                f"Error: Failed to download model from {ckpt_url} to {local_path}")
            raise e
    else:
        raise NotImplementedError(
            f"Download model file {global_path} not supported")

    assert local_path.exists(), f"Missing checkpoint {local_path}"

    return local_path.as_posix()
