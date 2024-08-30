from typing import List
from pathlib import Path
from modules.params.params_mixin import AsDictMixin


class InferenceParams(AsDictMixin):
    def __init__(self,
                 # reset seed (only for inference) at every start
                 reset_seed_per_generation: bool = True,
                 ):
        super().__init__()
        self.reset_seed_per_generation = reset_seed_per_generation

class T2VInferenceParams(InferenceParams):
    def __init__(self,
                 n_autoregressive_generations: int = 1,
                 num_conditional_frames: int = 8,  # during GENERATION, take the last frames,i.e. [:-num_conditional_frames]
                 # can be "15", i.e. take always the 16th frame of the entire video, or a range "-8:-1", take always frames -8:-1 of the last chunk
                 anchor_frames: str = "15",
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.n_autoregressive_generations = n_autoregressive_generations
        self.num_conditional_frames = num_conditional_frames
        self.anchor_frames = anchor_frames
