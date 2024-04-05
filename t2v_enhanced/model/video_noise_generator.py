import torch
import torch.fft as fft
from torch import nn
from torch.nn import functional
from math import sqrt
from einops import rearrange
import math
import numbers
from typing import List

# adapted from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
# and https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/19


def gaussian_smoothing_kernel(shape, kernel_size, sigma, dim=2):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32)
            for size in kernel_size
        ]
    )

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2

        kernel *= torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        # kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
        #    torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    pad_length = (math.floor(
        (shape[-1]-kernel_size[-1])/2), math.floor((shape[-1]-kernel_size[-1])/2), math.floor((shape[-2]-kernel_size[-2])/2), math.floor((shape[-2]-kernel_size[-2])/2), math.floor((shape[-3]-kernel_size[-3])/2), math.floor((shape[-3]-kernel_size[-3])/2))

    kernel = functional.pad(kernel, pad_length)
    assert kernel.shape == shape[-3:]
    return kernel

    '''
    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    
    self.register_buffer('weight', kernel)
    self.groups = channels

    if dim == 1:
        self.conv = functional.conv1d
    elif dim == 2:
        self.conv = functional.conv2d
    elif dim == 3:
        self.conv = functional.conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                dim)
        )
    '''


class NoiseGenerator():

    def __init__(self, alpha: float = 0.0, shared_noise_across_chunks: bool = False, mode="vanilla", forward_steps: int = 850, radius: List[float] = None) -> None:
        self.mode = mode
        self.alpha = alpha
        self.shared_noise_across_chunks = shared_noise_across_chunks
        self.forward_steps = forward_steps
        self.radius = radius

    def set_seed(self, seed: int):
        self.seed = seed

    def reset_seed(self, seed: int):
        pass

    def reset_noise_generator_state(self):
        if hasattr(self, "e_shared"):
            del self.e_shared

    def sample_noise(self, z_0: torch.tensor = None, shape=None, device=None, dtype=None, generator=None, content=None):
        assert (z_0 is not None) != (
            shape is not None), f"either z_0 must be None, or shape must be None. Both provided."
        kwargs = {}
        noise = torch.randn(shape, **kwargs)

        if z_0 is None:
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype

        else:
            kwargs["device"] = z_0.device
            kwargs["dtype"] = z_0.dtype
            shape = z_0.shape

        if generator is not None:
            kwargs["generator"] = generator

        B, F, C, W, H = shape
        if F == 4 and C > 4:
            frame_idx = 2
            F, C = C, F
        else:
            frame_idx = 1

        if "mixed_noise" in self.mode:

            shape_per_frame = [dim for dim in shape]
            shape_per_frame[frame_idx] = 1
            zero_mean = torch.zeros(
                shape_per_frame, device=kwargs["device"], dtype=kwargs["dtype"])
            std = torch.ones(
                shape_per_frame, device=kwargs["device"], dtype=kwargs["dtype"])
            alpha = self.alpha
            std_coeff_shared = (alpha**2) / (1 + alpha**2)
            if self.shared_noise_across_chunks and hasattr(self, "e_shared"):
                e_shared = self.e_shared
            else:
                e_shared = torch.normal(mean=zero_mean, std=sqrt(
                    std_coeff_shared)*std, generator=kwargs["generator"] if "generator" in kwargs else None)
                if self.shared_noise_across_chunks:
                    self.e_shared = e_shared

            e_inds = []
            for frame in range(shape[frame_idx]):
                std_coeff_ind = 1 / (1 + alpha**2)
                e_ind = torch.normal(
                    mean=zero_mean, std=sqrt(std_coeff_ind)*std, generator=kwargs["generator"] if "generator" in kwargs else None)
                e_inds.append(e_ind)
            noise = torch.cat(
                [e_shared + e_ind for e_ind in e_inds], dim=frame_idx)

        if "consistI2V" in self.mode and content is not None:
            # if self.mode == "mixed_noise_consistI2V", we will use 'noise' from 'mixed_noise'. Otherwise, it is randn noise.

            if frame_idx == 1:
                assert content.shape[0] == noise.shape[0] and content.shape[2:] == noise.shape[2:]
                content = torch.concat([content, content[:, -1:].repeat(
                    1, noise.shape[1]-content.shape[1], 1, 1, 1)], dim=1)
                noise = rearrange(noise, "B F C W H -> (B C) F W H")
                content = rearrange(content, "B F C W H -> (B C) F W H")

            else:
                assert content.shape[:2] == noise.shape[:
                                                        2] and content.shape[3:] == noise.shape[3:]
                content = torch.concat(
                    [content, content[:, :, -1:].repeat(1, 1, noise.shape[2]-content.shape[2], 1, 1)], dim=2)
                noise = rearrange(noise, "B C F W H -> (B C) F W H")
                content = rearrange(content, "B C F W H -> (B C) F W H")

            # TODO implement DDPM_forward using diffusers framework
            '''
            content_noisy = ddpm_forward(
                content, noise, self.forward_steps)
            '''

            # A 2D low pass filter was given in the blog:
            # see https://pytorch.org/blog/the-torch.fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pyTorch/

            # alternative
            # do we have to specify more (s,dim,norm?)
            noise_fft = fft.fftn(noise)
            content_noisy_fft = fft.fftn(content_noisy)

            # shift low frequency parts to center
            noise_fft_shifted = fft.fftshift(noise_fft)
            content_noisy_fft_shifted = fft.fftshift(content_noisy_fft)

            # create gaussian low pass filter 'gaussian_low_pass_filter' (specify std!)
            # mask out high frequencies using 'cutoff_frequence', something like gaussian_low_pass_filter[freq > cut_off_frequency] = 0.0
            # TODO define 'gaussian_low_pass_filter', apply frequency cutoff filter using self.cutoff_frequency. We need to apply fft.fftshift too probably.
            # TODO what exactly is the "normalized space-time stop frequency" used for the cutoff?

            gaussian_3d = gaussian_smoothing_kernel(noise_fft.shape, kernel_size=(
                noise_fft.shape[-3], noise_fft.shape[-2], noise_fft.shape[-1]), sigma=1, dim=3).to(noise.device)

            # define cutoff frequency around the kernel center
            # TODO define center and cut off radius, e.g. somethink like gaussian_3d[...,:c_x-r_x,:c_y-r_y:,:c_z-r_z] = 0.0 and gaussian_3d[...,c_x+r_x:,c_y+r_y:,c_z+r_z:] = 0.0
            # as we have 16 x 32 x 32, center should be (7.5,15.5,15.5)
            radius = self.radius

            # TODO we need to use rounding (ceil?)

            gaussian_3d[:center[0]-radius[0], :center[1] -
                        radius[1], :center[2]-radius[2]] = 0.0
            gaussian_3d[center[0]+radius[0]:,
                        center[1]+radius[1]:, center[2]+radius[2]:] = 0.0

            noise_fft_shifted_hp = noise_fft_shifted * (1 - gaussian_3d)
            content_noisy_fft_shifted_lp = content_noisy_fft_shifted * gaussian_3d

            noise = fft.ifftn(fft.ifftshift(
                noise_fft_shifted_hp+content_noisy_fft_shifted_lp))
            if frame_idx == 1:
                noise = rearrange(
                    noise, "(B C) F W H -> B F C W H", B=B)
            else:
                noise = rearrange(
                    noise, "(B C) F W H -> B C F W H", B=B)

        assert noise.shape == shape
        return noise
