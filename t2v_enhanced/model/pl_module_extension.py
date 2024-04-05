import torch
from copy import deepcopy
from einops import repeat
import math


class FrameConditioning():
    def __init__(self,
                 add_frame_to_input: bool = False,
                 add_frame_to_layers: bool = False,
                 fill_zero: bool = False,
                 randomize_mask: bool = False,
                 concatenate_mask: bool = False,
                 injection_probability: float = 0.9,
                 ) -> None:
        self.use = None
        self.add_frame_to_input = add_frame_to_input
        self.add_frame_to_layers = add_frame_to_layers
        self.fill_zero = fill_zero
        self.randomize_mask = randomize_mask
        self.concatenate_mask = concatenate_mask
        self.injection_probability = injection_probability
        self.add_frame_to_input or self.add_frame_to_layers

        assert not add_frame_to_layers or not add_frame_to_input

    def set_random_mask(self, random_mask: bool):
        frame_conditioning = deepcopy(self)
        frame_conditioning.randomize_mask = random_mask
        return frame_conditioning

    @property
    def use(self):
        return self.add_frame_to_input or self.add_frame_to_layers

    @use.setter
    def use(self, value):
        if value is not None:
            raise NotImplementedError("Direct access not allowed")

    def attach_video_frames(self, pl_module, z_0: torch.Tensor = None, batch: torch.Tensor = None, random_mask: bool = False):
        assert self.fill_zero, "Not filling with zero not implemented yet"
        n_frames_inference = self.inference_params.video_length
        with torch.no_grad():
            if z_0 is None:
                assert batch is not None
                z_0 = pl_module.encode_frame(batch)
            assert n_frames_inference == z_0.shape[1], "For frame injection, the number of frames sampled by the dataloader must match the number of frames used for video generation"
            shape = list(z_0.shape)

            shape[1] = pl_module.inference_params.video_length
            M = torch.zeros(shape, dtype=z_0.dtype,
                            device=pl_module.device)  # [B F C W H]
            bsz = z_0.shape[0]
            if random_mask:
                p_inject_frame = self.injection_probability
                use_masks = torch.bernoulli(
                    torch.tensor(p_inject_frame).repeat(bsz)).long()
                keep_frame_idx = torch.randint(
                    0, n_frames_inference, (bsz,), device=pl_module.device).long()
            else:
                use_masks = torch.ones((bsz,), device=pl_module.device).long()
                # keep only first frame
                keep_frame_idx = 0 * use_masks
            frame_idx = []

            for batch_idx, (keep_frame, use_mask) in enumerate(zip(keep_frame_idx, use_masks)):
                M[batch_idx, keep_frame] = use_mask
                frame_idx.append(keep_frame if use_mask == 1 else -1)

            x0 = z_0*M
            if self.concatenate_mask:
                # flatten mask
                M = M[:, :, 0, None]
                x0 = torch.cat([x0, M], dim=2)
            if getattr(pl_module.opt_params.noise_decomposition, "use", False) and random_mask:
                assert x0.shape[0] == 1, "randomizing frame injection with noise decomposition not implemented for batch size >1"
        return x0, frame_idx


class NoiseDecomposition():

    def __init__(self,
                 use: bool = False,
                 random_frame: bool = False,
                 lambda_f: float = 0.5,
                 use_base_model: bool = True,
                 ):
        self.use = use
        self.random_frame = random_frame
        self.lambda_f = lambda_f
        self.use_base_model = use_base_model

    def get_loss(self, x0, unet_base, unet, noise_scheduler, frame_idx, z_t_base, timesteps, encoder_hidden_states, base_noise, z_t_residual, composed_noise):
        if x0 is not None:
            # x0.shape = [B,F,C,W,H], if extrapolation_params.fill_zero=true, only one frame per batch non-zero
            assert not self.random_frame

            # TODO add x0 injection
            x0_base = []
            for batch_idx, frame in enumerate(frame_idx):
                x0_base.append(x0[batch_idx, frame, None, None])

            x0_base = torch.cat(x0_base, dim=0)
            x0_residual = repeat(
                x0[:, 0], "B C W H -> B F C W H", F=x0.shape[1]-1)
        else:
            x0_residual = None

        if self.use_base_model:
            base_pred = unet_base(z_t_base, timesteps,
                                  encoder_hidden_states, x0=x0_base).sample
        else:
            base_pred = base_noise

        timesteps_alphas = [
            noise_scheduler.alphas_cumprod[t.cpu()] for t in timesteps]
        timesteps_alphas = torch.stack(
            timesteps_alphas).to(base_pred.device)
        timesteps_alphas = repeat(timesteps_alphas, "B -> B F C W H",
                                  F=base_pred.shape[1], C=base_pred.shape[2], W=base_pred.shape[3], H=base_pred.shape[4])
        base_correction = math.sqrt(
            lambda_f) * torch.sqrt(1-timesteps_alphas) * base_pred

        z_t_residual_dash = z_t_residual - base_correction

        residual_pred = unet(
            z_t_residual_dash, timesteps, encoder_hidden_states, x0=x0_residual).sample
        composed_pred = math.sqrt(
            lambda_f)*base_pred.detach() + math.sqrt(1-lambda_f) * residual_pred

        loss_residual = torch.nn.functional.mse_loss(
            composed_noise.float(), composed_pred.float(), reduction=reduction)
        if self.use_base_model:
            loss_base = torch.nn.functional.mse_loss(
                base_noise.float(), base_pred.float(), reduction=reduction)
            loss = loss_residual+loss_base
        else:
            loss = loss_residual
        return loss

    def add_noise(self, z_base, base_noise, z_residual, composed_noise, noise_scheduler, timesteps):
        z_t_base = noise_scheduler.add_noise(
            z_base, base_noise, timesteps)
        z_t_residual = noise_scheduler.add_noise(
            z_residual, composed_noise, timesteps)
        return z_t_base, z_t_residual

    def split_latent_into_base_residual(self, z_0, pl_module, noise_generator):
        if self.random_frame:
            raise NotImplementedError("Must be synced with x0 mask!")
            fr_select = torch.randint(
                0, z_0.shape[1], (bsz,), device=pl_module.device).long()
            z_base = z_0[:, fr_Select, None]
            fr_residual = [fr for fr in range(
                z_0.shape[1]) if fr != fr_select]
            z_residual = z_0[:, fr_residual, None]
        else:
            if not pl_module.unet_params.frame_conditioning.randomize_mask:
                z_base = z_0[:, 0, None]
                z_residual = z_0[:, 1:]
            else:
                z_base = []
                for batch_idx, frame_at_batch in enumerate(frame_idx):
                    z_base.append(
                        z_0[batch_idx, frame_at_batch, None, None])
                z_base = torch.cat(z_base, dim=0)
            # z_residual = z_0[[:, 1:]
                z_residual = []

                for batch_idx, frame_idx_batch in enumerate(frame_idx):
                    z_residual_batch = []
                    for frame in range(z_0.shape[1]):
                        if frame_idx_batch != frame:
                            z_residual_batch.append(
                                z_0[batch_idx, frame, None, None])
                    z_residual_batch = torch.cat(
                        z_residual_batch, dim=1)
                    z_residual.append(z_residual_batch)
                z_residual = torch.cat(z_residual, dim=0)
        base_noise = noise_generator.sample_noise(z_base)  # b_t
        residual_noise = noise_generator.sample_noise(z_residual)  # r^f_t
        lambda_f = self.lambda_f
        composed_noise = math.sqrt(
            lambda_f) * base_noise + math.sqrt(1-lambda_f) * residual_noise  # dimension issue?

        return z_base, base_noise, z_residual, composed_noise


class NoiseGenerator():

    def __init__(self, mode="vanilla") -> None:
        self.mode = mode

    def set_seed(self, seed: int):
        self.seed = seed

    def reset_seed(self, seed: int):
        pass

    def sample_noise(self, z_0: torch.tensor = None, shape=None, device=None, dtype=None, generator=None):

        assert (z_0 is not None) != (
            shape is not None), f"either z_0 must be None, or shape must be None. Both provided."
        kwargs = {}

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

        if self.mode == "vanilla":
            noise = torch.randn(
                shape, **kwargs)
        elif self.mode == "free_noise":
            noise = torch.randn(shape, **kwargs)
            if noise.shape[1] > 4:
                # HARD CODED
                noise = noise[:, :8]
                noise = torch.cat(
                    [noise, noise[:, torch.randperm(noise.shape[1])]], dim=1)
            elif noise.shape[2] > 4:
                noise = noise[:, :, :8]
                noise = torch.cat(
                    [noise, noise[:, :, torch.randperm(noise.shape[2])]], dim=2)
            else:
                raise NotImplementedError(
                    f"Shape of noise vector not as expected {noise.shape}")
        elif self.mode == "equal":
            shape = list(shape)
            shape[1] = 1
            noise_init = torch.randn(
                shape, **kwargs)
            shape[1] = F
            noise = torch.zeros(
                shape, device=noise_init.device, dtype=noise_init.dtype)
            for fr in range(F):
                noise[:, fr] = noise_init[:, 0]
        elif self.mode == "fusion":
            shape = list(shape)
            shape[1] = 1
            noise_init = torch.randn(
                shape, **kwargs)
            noises = []
            noises.append(noise_init)
            for fr in range(F-1):

                shift = 2*(fr+1)
                local_copy = noise_init
                shifted_noise = torch.cat(
                    [local_copy[:, :, :, shift:, :], local_copy[:, :, :, :shift, :]], dim=3)
                noises.append(math.sqrt(0.2)*shifted_noise +
                              math.sqrt(1-0.2)*torch.rand(shape, **kwargs))
            noise = torch.cat(noises, dim=1)

        elif self.mode == "motion_dynamics" or self.mode == "equal_noise_per_sequence":

            shape = list(shape)
            normal_frames = 1
            shape[1] = normal_frames
            init_noise = torch.randn(
                shape, **kwargs)
            noises = []
            noises.append(init_noise)
            init_noise = init_noise[:, -1, None]
            print(f"UPDATE with noise = {init_noise.shape}")

            if self.mode == "motion_dynamics":
                for fr in range(F-normal_frames):

                    shift = 2*(fr+1)
                    print(fr, shift)
                    local_copy = init_noise
                    shifted_noise = torch.cat(
                        [local_copy[:, :, :, shift:, :], local_copy[:, :, :, :shift, :]], dim=3)
                    noises.append(shifted_noise)
            elif self.mode == "equal_noise_per_sequence":
                for fr in range(F-1):
                    noises.append(init_noise)
            else:
                raise NotImplementedError()
            # noises[0] = noises[0] * 0
            noise = torch.cat(noises, dim=1)
            print(noise.shape)

        return noise
