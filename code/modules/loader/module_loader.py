from diffusers import DDPMScheduler, DiffusionPipeline
from typing import List, Any, Union, Type
from utils.loader import get_class
from copy import deepcopy
from modules.loader.module_loader_config import ModuleLoaderConfig
import torch
import pytorch_lightning as pl
import jsonargparse


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class GenericModuleLoader():

    def __init__(self,
                 pipeline_repo: str = None,
                 pipeline_obj: str = None,
                 set_prediction_type: str = "",
                 module_names: List[str] = [
                     "scheduler", "text_encoder", "tokenizer", "vae", "unet",],
                 module_config: dict[str,
                                     Union[ModuleLoaderConfig,  torch.nn.Module, Any]] = None,
                 fast_dev_run: Union[int, bool] = False,
                 root_cls: Type[Any] = None,
                 ) -> None:
        self.module_config = module_config
        self.pipeline_repo = pipeline_repo
        self.pipeline_obj = pipeline_obj
        self.set_prediction_type = set_prediction_type
        self.module_names = module_names
        self.fast_dev_run = fast_dev_run
        self.root_cls = root_cls

    def load_custom_scheduler(self):
        module_obj = DDPMScheduler.from_pretrained(
            self.pipeline_repo, subfolder="scheduler")

        if len(self.set_prediction_type) > 0:
            scheduler_config = module_obj.load_config(
                self.pipeline_repo, subfolder="scheduler")
            scheduler_config["prediction_type"] = self.set_prediction_type
            module_obj = module_obj.from_config(scheduler_config)
        return module_obj

    def load_pipeline(self):
        return DiffusionPipeline.from_pretrained(self.pipeline_repo) if self.pipeline_repo is not None else None

    def __call__(self, trainer: pl.LightningModule, diff_trainer_params):
        # load diffusers pipeline object if set
        if self.pipeline_obj is not None:
            pipe = self.load_pipeline()
        else:
            pipe = None

        if pipe is not None and self.pipeline_obj is not None:
            # store the entire diffusers pipeline object under the name given by pipeline_obj
            setattr(trainer, self.pipeline_obj, self.load_pipeline())

        for module_name in self.module_names:
            print(f" --- START:  Loading module: {module_name} ---")
            if module_name not in self.module_config.keys() and pipe is not None:
                # stores models from already loaded diffusers pipeline
                module_obj = getattr(pipe, module_name)
                if module_name == "scheduler":
                    module_obj = self.load_custom_scheduler()
                setattr(trainer, module_name, module_obj)
            else:
                if not isinstance(self.module_config[module_name], ModuleLoaderConfig):
                    # instantiate model by jsonargparse and store it
                    module = self.module_config[module_name]
                    # TODO we want to be able to load ckpt still.
                    config_obj = None
                else:
                    # instantiate object from class method (as used by Diffusers, e.g. DiffusionPipeline.load_from_pretrained)
                    config_obj = self.module_config[module_name]
                    # retrieve loader class
                    loader_cls = get_class(
                        config_obj.loader_cls_path)

                    # retrieve loader method
                    if config_obj.cls_func != "":
                        # we allow to specify a method for fast loading (e.g. in diffusers, from_config instead of from_pretrained)
                        # makes loading faster for quick testing
                        if not self.fast_dev_run or config_obj.cls_func_fast_dev_run == "":
                            cls_func = getattr(
                                loader_cls, config_obj.cls_func)
                        else:
                            print(
                                f"Model {module_name}: loading fast_dev_run class loader")
                            cls_func = getattr(
                                loader_cls, config_obj.cls_func_fast_dev_run)
                    else:
                        cls_func = loader_cls

                    # retrieve parameters
                    # load parameters specified in diff_trainer_params (so it links them)
                    kwargs_trainer_params = config_obj.kwargs_diff_trainer_params

                    kwargs_diffusers = config_obj.kwargs_diffusers

                    # names of dependent modules that we need as input
                    dependent_modules = config_obj.dependent_modules

                    # names of dependent modules that we need as input. Modules will be cloned
                    dependent_modules_cloned = config_obj.dependent_modules_cloned

                    # model kwargs. Can be just a dict, or a parameter class (derived from modules.params.params_mixin.AsDictMixin) so we have verification of inputs
                    model_params = config_obj.model_params

                    # kwargs used only if on fast_dev_run mode
                    model_params_fast_dev_run = config_obj.model_params_fast_dev_run

                    if model_params is not None:
                        if isinstance(model_params, dict):
                            model_dict = model_params
                        else:
                            model_dict = model_params.to_dict()
                    else:
                        model_dict = {}

                    if (model_params_fast_dev_run is None) or (not self.fast_dev_run):
                        model_params_fast_dev_run = {}
                    else:
                        print(
                            f"Module {module_name}: loading fast_dev_run params")

                    loaded_modules_dict = {}
                    if dependent_modules is not None:
                        for key, dependent_module in dependent_modules.items():
                            assert hasattr(
                                trainer, dependent_module), f"Module {dependent_module} not available. Set {dependent_module} before module {module_name} in module_loader.module_names. Current order: {self.module_names}"
                            loaded_modules_dict[key] = getattr(
                                trainer, dependent_module)

                    if dependent_modules_cloned is not None:
                        for key, dependent_module in dependent_modules_cloned.items():
                            assert hasattr(
                                trainer, dependent_module), f"Module {dependent_module} not available. Set {dependent_module} before module {module_name} in module_loader.module_names. Current order: {self.module_names}"
                            loaded_modules_dict[key] = getattr(
                                trainer, deepcopy(dependent_module))
                    if kwargs_trainer_params is not None:
                        for key, param in kwargs_trainer_params.items():
                            if param is not None:
                                kwargs_trainer_params[key] = getattr(
                                    diff_trainer_params, param)
                            else:
                                kwargs_trainer_params[key] = diff_trainer_params
                    else:
                        kwargs_trainer_params = {}

                    if kwargs_diffusers is None:
                        kwargs_diffusers = {}
                    else:
                        for key, value in kwargs_diffusers.items():
                            if key == "torch_dtype":
                                if value == "torch.float16":
                                    kwargs_diffusers[key] = torch.float16

                    kwargs = kwargs_diffusers | loaded_modules_dict | kwargs_trainer_params | model_dict | model_params_fast_dev_run
                    args = config_obj.args
                    # instantiate object
                    module = cls_func(*args, **kwargs)
                    module: torch.nn.Module
                    if self.root_cls is not None:
                        assert isinstance(module, self.root_cls)

                if config_obj is not None and config_obj.state_dict_path != "" and not self.fast_dev_run:
                    # TODO extend loading to hf spaces
                    print(
                        f"             * Loading checkpoint {config_obj.state_dict_path} - STARTED")
                    module_state_dict = torch.load(
                        config_obj.state_dict_path, map_location=torch.device("cpu"))
                    module_state_dict = module_state_dict["state_dict"]

                    if len(config_obj.state_dict_filters) > 0:
                        assert not config_obj.strict_loading
                        ckpt_params_dict = {}
                        for name, param in module.named_parameters(prefix=module_name):
                            for filter_str in config_obj.state_dict_filters:
                                filter_groups = filter_str.split("*")
                                has_all_parts = True
                                for filter_group in filter_groups:
                                    has_all_parts = has_all_parts and filter_group in name

                                if has_all_parts:
                                    validate_name = name
                                    for filter_group in filter_groups:
                                        if filter_group in validate_name:
                                            shift = validate_name.index(
                                                filter_group)
                                            validate_name = validate_name[shift+len(
                                                filter_group):]
                                        else:
                                            has_all_parts = False
                                            break
                                if has_all_parts:
                                    ckpt_params_dict[name[len(
                                        module_name+"."):]] = param
                    else:
                        ckpt_params_dict = dict(filter(lambda x: x[0].startswith(
                            module_name), module_state_dict.items()))
                        ckpt_params_dict = {
                            k.split(module_name+".")[1]: v for (k, v) in ckpt_params_dict.items()}

                    if len(ckpt_params_dict) > 0:
                        miss, unex = module.load_state_dict(
                            ckpt_params_dict, strict=config_obj.strict_loading)
                        ckpt_params_dict = {}
                        assert len(
                            unex) == 0, f"Unexpected parameters in checkpoint: {unex}"
                        if len(miss) > 0:
                            print(
                                f"Checkpoint {config_obj.state_dict_path} is missing parameters for module {module_name}.")
                            print(miss)
                    print(
                        f"             * Loading checkpoint {config_obj.state_dict_path} - FINISHED")
                if isinstance(module, jsonargparse.Namespace) or isinstance(module, dict):
                    print(bcolors.WARNING +
                          f"Warning: Seems object {module_name} was not build correct." + bcolors.ENDC)

                setattr(trainer, module_name, module)
            print(f" --- FINSHED:  Loading module: {module_name} ---")
