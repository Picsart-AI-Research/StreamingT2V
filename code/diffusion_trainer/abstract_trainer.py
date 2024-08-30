import os

import pytorch_lightning as pl
import torch

from typing import Any

from modules.params.diffusion.inference_params import InferenceParams
from modules.loader.module_loader import GenericModuleLoader
from modules.params.diffusion_trainer.params_streaming_diff_trainer import DiffusionTrainerParams


class AbstractTrainer(pl.LightningModule):

    def __init__(self,
                 inference_params: Any,
                 diff_trainer_params: DiffusionTrainerParams,
                 module_loader: GenericModuleLoader,
                 ):

        super().__init__()

        self.inference_params = inference_params
        self.diff_trainer_params = diff_trainer_params
        self.module_loader = module_loader

        self.on_start_once_called = False
        self._setup_methods = []

        module_loader(
            trainer=self,
            diff_trainer_params=diff_trainer_params)

    # ------ IMPLEMENTATION HOOKS -------

    def post_init(self, batch):
        '''
        Is called after LightningDataModule and LightningModule is created, but before any training/validation/prediction. 
        First possible access to the 'trainer' object (e.g. to get 'device').  
        '''

    def generate_output(self, batch, batch_idx, inference_params: InferenceParams):
        '''
        Is called during validation to generate for each batch an output.
        Return the meta information about produced result (where result were stored).
        This is used for the metric evaluation.
        '''

    # ------- HELPER FUNCTIONS -------

    def _reset_random_generator(self):
        '''
        Reset the random generator to the same seed across all workers. The generator is used only for inference.
        '''
        if not hasattr(self, "random_generator"):
            self.random_generator = torch.Generator(device=self.device)
            # set seed according to 'seed_everything' in config
            seed = int(os.environ.get("PL_GLOBAL_SEED", 42))
        else:
            seed = self.random_generator.initial_seed()
        self.random_generator.manual_seed(seed)

    # ----- PREDICT HOOKS ------

    def on_predict_start(self):
        self.on_start()

    def predict_step(self, batch, batch_idx):
        self.on_inference_step(batch=batch, batch_idx=batch_idx)

    def on_predict_epoch_start(self):
        self.on_inference_epoch_start()

    # -----   CUSTOM HOOKS -----

    # Global Hooks (Called by Training, Validation and Prediction)

    # abstract method

    def _on_start_once(self):
        '''
        Will be called only once by on_start. Thus, it will be called by the first call of train,validation or prediction. 
        '''
        if self.on_start_once_called:
            return
        else:
            self.on_start_once_called = True
        self.post_init()

    def on_start(self):
        '''
        Called at the beginning of training, validation and prediction.
        '''
        self._on_start_once()

    # Inference Hooks (Called by Validation and Prediction)

     # ----- Inference Hooks (called by 'validation' and 'predict') ------

    def on_inference_epoch_start(self):
        # reset seed at every inference
        self._reset_random_generator()

    def on_inference_step(self, batch, batch_idx):
        if self.inference_params.reset_seed_per_generation:
            self._reset_random_generator()
        self.generate_output(
            batch=batch, inference_params=self.inference_params, batch_idx=batch_idx)
