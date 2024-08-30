from typing import Any, Union, List


class ModuleLoaderConfig:

    def __init__(self,
                 loader_cls_path: str,
                 cls_func: str = "",
                 cls_func_fast_dev_run: str = "",
                 kwargs_diffusers: dict[str, Any] = None,
                 # model kwargs. Can be just a dict, or a parameter class (derived from modules.params.params_mixin.AsDictMixin) so we have verification of inputs
                 model_params: Any = None,
                 # kwargs activated only if on fast_dev_run mode
                 model_params_fast_dev_run: Any = None,
                 # load parameters specified in diff_trainer_params (so it links them)
                 kwargs_diff_trainer_params: dict[str,
                                                  Union[str, None]] = None,
                 args: List[Any] = [],
                 # names of dependent modules that we need as input
                 dependent_modules: dict[str, str] = None,
                 # names of dependent modules that we need as input. Modules will be cloned
                 dependent_modules_cloned: dict[str, str] = None,
                 state_dict_path: str = "",
                 strict_loading: bool = True,
                 state_dict_filters: List[str] = []
                 ) -> None:
        self.loader_cls_path = loader_cls_path
        self.cls_func = cls_func
        self.cls_func_fast_dev_run = cls_func_fast_dev_run
        self.kwargs_diffusers = kwargs_diffusers
        self.dependent_modules = dependent_modules
        self.dependent_modules_cloned = dependent_modules_cloned
        self.kwargs_diff_trainer_params = kwargs_diff_trainer_params
        self.model_params = model_params
        self.state_dict_path = state_dict_path
        self.strict_loading = strict_loading
        self.state_dict_filters = state_dict_filters
        self.model_params_fast_dev_run = model_params_fast_dev_run
        self.args = args
