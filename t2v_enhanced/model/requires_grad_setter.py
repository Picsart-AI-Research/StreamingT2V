from typing import Union, Any, Dict, List, Optional, Tuple
import pytorch_lightning as pl


class LayerConfig():
    def __init__(self,
                 gradient_setup: List[Tuple[bool, List[str]]] = None,
                 ) -> None:

        if gradient_setup is not None:
            self.gradient_setup = gradient_setup
        self.new_config = True
        # TODO add option to specify quantization per layer

    def set_requires_grad(self, pl_module: pl.LightningModule):
        # [["True","unet.a.b","c"],["True,[]"]]

        for selected_module_setup in self.gradient_setup:
            for model_name, p in pl_module.named_parameters():
                grad_mode = selected_module_setup[0] == True
                selected_module_path = selected_module_setup[1]
                path_is_matching = True
                model_name_selection = model_name
                for selected_module in selected_module_path:
                    position = model_name_selection.find(selected_module)
                    if position == -1:
                        path_is_matching = False
                        continue
                    else:
                        shift = len(selected_module)
                        model_name_selection = model_name_selection[position+shift:]
                if path_is_matching:
                    # if grad_mode:
                    # print(
                    #    f"Setting gradient for {model_name} to {grad_mode}")
                    p.requires_grad = grad_mode
