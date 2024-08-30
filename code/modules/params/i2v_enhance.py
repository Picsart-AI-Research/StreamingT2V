from modules.params.params_mixin import AsDictMixin


class I2VEnhanceParams(AsDictMixin):

    def __init__(self,
                 ckpt_path_local: str = "",
                 ckpt_path_global: str = "",
                 ) -> None:
        super().__init__()
        self.ckpt_path_local = ckpt_path_local
        self.ckpt_path_global = ckpt_path_global
