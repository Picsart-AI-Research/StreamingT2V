from modules.params.params_mixin import AsDictMixin


class CheckpointDescriptor():

    def __init__(self,
                 ckpt_path_local: str = None,
                 ckpt_path_global: str = None):
        self.ckpt_path_local = ckpt_path_local
        self.ckpt_path_global = ckpt_path_global


class DiffusionTrainerParams(AsDictMixin):

    def __init__(self,
                 scale_factor: float = 0.18215,
                 streamingsvd_ckpt: CheckpointDescriptor = None,
                 disable_first_stage_autocast: bool = False,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.streamingsvd_ckpt = streamingsvd_ckpt
        self.disable_first_stage_autocast = disable_first_stage_autocast
