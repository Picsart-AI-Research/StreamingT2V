# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Part of the implementation is borrowed and modified from ControlNet,
# publicly available at https://github.com/lllyasviel/ControlNet
import torch


class BaseModel(torch.nn.Module):

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if 'optimizer' in parameters:
            parameters = parameters['model']

        self.load_state_dict(parameters)
