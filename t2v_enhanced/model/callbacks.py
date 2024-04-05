
from pathlib import Path
from pytorch_lightning import Callback
import os
import torch
from lightning_fabric.utilities.cloud_io import get_filesystem
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning import LightningModule, Trainer
from lightning_utilities.core.imports import RequirementCache
from omegaconf import OmegaConf

_JSONARGPARSE_SIGNATURES_AVAILABLE = RequirementCache(
    "jsonargparse[signatures]>=4.17.0")

if _JSONARGPARSE_SIGNATURES_AVAILABLE:
    import docstring_parser
    from jsonargparse import (
        ActionConfigFile,
        ArgumentParser,
        class_from_function,
        Namespace,
        register_unresolvable_import_paths,
        set_config_read_mode,
    )

    # Required until fix https://github.com/pytorch/pytorch/issues/74483
    register_unresolvable_import_paths(torch)
    set_config_read_mode(fsspec_enabled=True)
else:
    locals()["ArgumentParser"] = object
    locals()["Namespace"] = object


class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        multifile: When input is multiple config files, saved config preserves this structure.

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run
    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        log_dir: str,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,

    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.already_saved = False
        self.log_dir = log_dir

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        log_dir = self.log_dir
        assert log_dir is not None
        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(
                config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            self.already_saved = True
            trainer.logger.log_hyperparams(OmegaConf.load(config_path))

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)
