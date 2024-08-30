import numpy as np
import torch

from models.svd.sgm.modules.diffusionmodules.discretizer import Discretization


# Implementation of https://arxiv.org/abs/2404.14507
class AlignYourSteps(Discretization):

    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def loglinear_interp(self, t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        xs = np.linspace(0, 1, len(t_steps))
        ys = np.log(t_steps[::-1])

        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)

        interped_ys = np.exp(new_ys)[::-1].copy()
        return interped_ys

    def get_sigmas(self, n, device="cpu"):
        sampling_schedule = [700.00, 54.5, 15.886, 7.977,
                             4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002]
        sigmas = torch.from_numpy(self.loglinear_interp(
            sampling_schedule, n)).to(device)
        return sigmas
