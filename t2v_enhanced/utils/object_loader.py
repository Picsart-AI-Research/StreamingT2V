import importlib
from functools import partialmethod


def instantiate_object(cls_path: str, *args, **kwargs):
    class_ = get_class(cls_path, *args, **kwargs)
    obj = class_()
    return obj


def get_class(cls_path: str, *args, **kwargs):
    module_name = ".".join(cls_path.split(".")[:-1])
    module = importlib.import_module(module_name)

    class_ = getattr(module, cls_path.split(".")[-1])
    class_.__init__ = partialmethod(class_.__init__, *args, **kwargs)
    return class_


if __name__ == "__main__":

    class_ = get_class(
        "diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler")
    scheduler = class_.from_config("stabilityai/stable-diffusion-2-1",
                                   subfolder="scheduler")
    print(scheduler)
