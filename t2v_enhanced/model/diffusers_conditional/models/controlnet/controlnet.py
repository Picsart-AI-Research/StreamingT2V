# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
# from diffusers.models.transformer_temporal import TransformerTemporalModel
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from t2v_enhanced.model.diffusers_conditional.models.controlnet.unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
    transformer_g_c
)
# from diffusers.models.unet_3d_condition import UNet3DConditionModel
from t2v_enhanced.model.diffusers_conditional.models.controlnet.unet_3d_condition import UNet3DConditionModel
from t2v_enhanced.model.diffusers_conditional.models.controlnet.transformer_temporal import TransformerTemporalModel
from t2v_enhanced.model.layers.conv_channel_extension import Conv2D_SubChannels
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetOutput(BaseOutput):
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class Merger(nn.Module):
    def __init__(self, n_frames_condition: int = 8, n_frames_sample: int = 16, merge_mode: str = "addition", input_channels=0, frame_expansion="last_frame") -> None:
        super().__init__()
        self.merge_mode = merge_mode
        self.n_frames_condition = n_frames_condition
        self.n_frames_sample = n_frames_sample
        self.frame_expansion = frame_expansion

        if merge_mode.startswith("attention"):
            self.attention = ConditionalModel(input_channels=input_channels,
                                              conditional_model=merge_mode.split("attention_")[1])

    def forward(self, x, condition_signal):
        x = rearrange(x, "(B F) C H W -> B F C H W", F=self.n_frames_sample)

        condition_signal = rearrange(
            condition_signal, "(B F) C H W -> B F C H W", B=x.shape[0])

        if x.shape[1] - condition_signal.shape[1] > 0:
            if self.frame_expansion == "last_frame":
                fillup_latent = repeat(
                    condition_signal[:, -1], "B C H W -> B F C H W", F=x.shape[1] - condition_signal.shape[1])
            elif self.frame_expansion == "zero":
                fillup_latent = torch.zeros(
                    (x.shape[0], self.n_frames_sample-self.n_frames_condition, *x.shape[2:]), device=x.device, dtype=x.dtype)

            if self.frame_expansion != "none":
                condition_signal = torch.cat(
                    [condition_signal, fillup_latent], dim=1)

        if self.merge_mode == "addition":
            out = x + condition_signal
        elif self.merge_mode.startswith("attention"):
            out = self.attention(x, condition_signal)
        out = rearrange(out, "B F C H W -> (B F) C H W")
        return out


class ZeroConv(nn.Module):
    def __init__(self, channels: int, mode: str = "2d", num_frames: int = 8, zero_init=True):
        super().__init__()
        mode_parts = mode.split("_")
        if len(mode_parts) > 1 and mode_parts[1] == "noinit":
            zero_init = False

        if mode.startswith("2d"):
            model = nn.Conv2d(
                channels, channels, kernel_size=1)
            model = zero_module(model, reset=zero_init)
        elif mode.startswith("3d"):
            model = ZeroConv3D(num_frames=num_frames,
                               channels=channels, zero_init=zero_init)
        elif mode == "Identity":
            model = nn.Identity()
        self.model = model

    def forward(self, x):
        return self.model(x)





class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """
    # TODO why not GAUSSIAN used?
    # TODO why not 4x4 kernel?
    # TODO why not 2 x2 stride?

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
        downsample: bool = True,
        final_3d_conv: bool = False,
        num_frame_conditioning: int = 8,
        num_frames: int = 16,
        zero_init: bool = True,
        use_controlnet_mask: bool = False,
        use_normalization: bool = False,
    ):
        super().__init__()
        self.num_frame_conditioning = num_frame_conditioning
        self.num_frames = num_frames
        self.final_3d_conv = final_3d_conv
        self.conv_in = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        if final_3d_conv:
            print("USING 3D CONV in ControlNET")

        self.blocks = nn.ModuleList([])
        if use_normalization:
            self.norms = nn.ModuleList([])
        self.use_normalization = use_normalization

        stride = 2 if downsample else 1
        if use_normalization:
            res = 256  # HARD-CODED Resolution!

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            if use_normalization:
                self.norms.append(nn.LayerNorm((channel_in, res, res)))
            self.blocks.append(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=stride))
            if use_normalization:
                res = res // 2
                self.norms.append(nn.LayerNorm((channel_out, res, res)))

        if not final_3d_conv:
            self.conv_out = zero_module(
                nn.Conv2d(
                    block_out_channels[-1]+int(use_controlnet_mask), conditioning_embedding_channels, kernel_size=3, padding=1), reset=zero_init
            )
        else:
            self.conv_temp = zero_module(TemporalConvLayer_Custom(
                num_frame_conditioning, num_frames, dropout=0.0), reset=zero_init)
            self.conv_out = nn.Conv2d(
                block_out_channels[-1]+int(use_controlnet_mask), conditioning_embedding_channels, kernel_size=3, padding=1)
           # self.conv_temp = zero_module(nn.Conv3d(
           #     num_frame_conditioning, num_frames, kernel_size=3, padding=1)
            # )

    def forward(self, conditioning, vq_gan=None, controlnet_mask=None):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        if self.use_normalization:
            for block, norm in zip(self.blocks, self.norms):
                embedding = block(embedding)
                embedding = norm(embedding)
                embedding = F.silu(embedding)
        else:
            for block in self.blocks:
                embedding = block(embedding)
                embedding = F.silu(embedding)

        if controlnet_mask is not None:
            embedding = rearrange(
                embedding, "(B F) C H W -> F B C H W", F=self.num_frames)
            controlnet_mask_expanded = controlnet_mask[:, :, None, None, None]
            controlnet_mask_expanded = rearrange(
                controlnet_mask_expanded, "B F C W H -> F B C W H")
            masked_embedding = controlnet_mask_expanded * embedding
            embedding = rearrange(masked_embedding, "F B C H W -> (B F) C H W")
            controlnet_mask_expanded = rearrange(
                controlnet_mask_expanded, "F B C H W -> (B F) C H W")
            # controlnet_mask_expanded = repeat(controlnet_mask_expanded,"B C W H -> B (C x) W H",x=embedding.shape[1])
            controlnet_mask_expanded = repeat(
                controlnet_mask_expanded, "B C W H -> B C (W y) H", y=embedding.shape[2])
            controlnet_mask_expanded = repeat(
                controlnet_mask_expanded, "B C W H -> B C W (H z)", z=embedding.shape[3])

            embedding = torch.cat([embedding, controlnet_mask_expanded], dim=1)

        embedding = self.conv_out(embedding)
        if self.final_3d_conv:
            # embedding = F.silu(embedding)
            embedding = rearrange(
                embedding, "(b f) c h w -> b f c h w", f=self.num_frame_conditioning)
            embedding = self.conv_temp(embedding)
            embedding = rearrange(embedding, "b f c h w -> (b f) c h w")

        return embedding

class ControlNetModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (
            16, 32, 96, 256),
        global_pool_conditions: bool = False,
        downsample_controlnet_cond: bool = True,
        frame_expansion: str = "zero",
        condition_encoder: str = "",
        num_frames: int = 16,
        num_frame_conditioning: int = 8,
        num_tranformers: int = 1,
        vae=None,
        merging_mode: str = "addition",
        zero_conv_mode: str = "2d",
        use_controlnet_mask: bool = False,
        use_image_embedding: bool = False,
        use_image_encoder_normalization: bool = False,
        unet_params=None,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )
        self.use_image_tokens = unet_params.use_image_tokens_ctrl
        self.image_encoder_name = type(unet_params.image_encoder).__name__

        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        '''Conv2D_SubChannels
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )
        '''
        self.conv_in = Conv2D_SubChannels(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )
        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(
            block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        self.transformer_in = TransformerTemporalModel(
            num_attention_heads=8,
            attention_head_dim=attention_head_dim,
            in_channels=block_out_channels[0],
            num_layers=1,
        )

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(
                num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(
                timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(
                projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None
        conditioning_channels = 3 if downsample_controlnet_cond else 4
        # control net conditioning embedding

        if condition_encoder == "temp_conv_vq":
            controlnet_cond_embedding = ControlNetConditioningEmbeddingVQ(
                conditioning_embedding_channels=block_out_channels[0],
                conditioning_channels=4,
                block_out_channels=conditioning_embedding_out_channels,
                downsample=False,

                num_frame_conditioning=num_frame_conditioning,
                num_frames=num_frames,
                num_tranformers=num_tranformers,
                # zero_init=not merging_mode.startswith("attention"),
            )
        elif condition_encoder == "vq":
            controlnet_cond_embedding = ControlNetConditioningOptVQ(vq=vae,
                                                                    conditioning_embedding_channels=block_out_channels[
                                                                        0],
                                                                    conditioning_channels=4,
                                                                    block_out_channels=conditioning_embedding_out_channels,
                                                                    num_frame_conditioning=num_frame_conditioning,
                                                                    num_frames=num_frames,
                                                                    )

        else:
            controlnet_cond_embedding = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=block_out_channels[0],
                conditioning_channels=conditioning_channels,
                block_out_channels=conditioning_embedding_out_channels,
                downsample=downsample_controlnet_cond,
                final_3d_conv=condition_encoder.endswith("3DConv"),
                num_frame_conditioning=num_frame_conditioning,
                num_frames=num_frames,
                # zero_init=not merging_mode.startswith("attention")
                use_controlnet_mask=use_controlnet_mask,
                use_normalization=use_image_encoder_normalization,
            )
        self.use_controlnet_mask = use_controlnet_mask
        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        # conv_in
        self.merger = Merger(n_frames_sample=num_frames, n_frames_condition=num_frame_conditioning,
                             merge_mode=merging_mode, input_channels=block_out_channels[0], frame_expansion=frame_expansion)

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [
                only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        self.controlnet_down_blocks.append(
            ZeroConv(channels=output_channel, mode=zero_conv_mode, num_frames=num_frames))
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=False,
                use_image_embedding=use_image_embedding,
                unet_params=unet_params,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                self.controlnet_down_blocks.append(
                    ZeroConv(channels=output_channel, mode=zero_conv_mode, num_frames=num_frames))

            if not is_final_block:
                self.controlnet_down_blocks.append(
                    ZeroConv(channels=output_channel, mode=zero_conv_mode, num_frames=num_frames))

        # mid
        mid_block_channel = block_out_channels[-1]

        self.controlnet_mid_block = ZeroConv(
            channels=mid_block_channel, mode=zero_conv_mode, num_frames=num_frames)

        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
            use_image_embedding=use_image_embedding,
            unet_params=unet_params,
        )
        self.controlnet_cond_embedding = controlnet_cond_embedding
        self.num_frames = num_frames
        self.num_frame_conditioning = num_frame_conditioning

    @classmethod
    def from_unet(
        cls,
        unet: UNet3DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (
            16, 32, 96, 256),
        load_weights_from_unet: bool = True,
        downsample_controlnet_cond: bool = True,
        num_frames: int = 16,
        num_frame_conditioning: int = 8,
        frame_expansion: str = "zero",
        num_tranformers: int = 1,
        vae=None,
        zero_conv_mode: str = "2d",
        merging_mode: str = "addition",
        # [spatial,spatial_3DConv,temp_conv_vq]
        condition_encoder: str = "spatial_3DConv",
        use_controlnet_mask: bool = False,
        use_image_embedding: bool = False,
        use_image_encoder_normalization: bool = False,
        unet_params=None,
        ** kwargs,
    ):
        r"""
        Instantiate Controlnet class from UNet3DConditionModel.

        Parameters:
            unet (`UNet3DConditionModel`):
                UNet model which weights are copied to the ControlNet. Note that all configuration options are also
                copied where applicable.
        """
        controlnet = cls(
            in_channels=unet.config.in_channels,
            down_block_types=unet.config.down_block_types,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            downsample_controlnet_cond=downsample_controlnet_cond,
            num_frame_conditioning=num_frame_conditioning,
            num_frames=num_frames,
            frame_expansion=frame_expansion,
            num_tranformers=num_tranformers,
            vae=vae,
            zero_conv_mode=zero_conv_mode,
            merging_mode=merging_mode,
            condition_encoder=condition_encoder,
            use_controlnet_mask=use_controlnet_mask,
            use_image_embedding=use_image_embedding,
            use_image_encoder_normalization=use_image_encoder_normalization,
            unet_params=unet_params,

        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.transformer_in.load_state_dict(
                unet.transformer_in.state_dict())
            controlnet.time_embedding.load_state_dict(
                unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(
                    unet.class_embedding.state_dict())

            controlnet.down_blocks.load_state_dict(
                unet.down_blocks.state_dict(), strict=False)  # can be that the controlnet model does not use image clip encoding
            controlnet.mid_block.load_state_dict(
                unet.mid_block.state_dict(), strict=False)

        return controlnet

    @property
    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(
                    f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(
                    f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * \
            [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(
                    f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D)):
            module.gradient_checkpointing = value

    # TODO ADD WEIGHT CONTROL
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
        weight_control: float = 1.0,
        weight_control_sample: float = 1.0,
        controlnet_mask: Optional[torch.Tensor] = None,
        vq_gan=None,
    ) -> Union[ControlNetOutput, Tuple]:
        # check channel order
        # TODO SET ATTENTION MASK And WEIGHT CONTROL as in CONTROLNET.PY
        '''
        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        '''
        # assert controlnet_mask is None, "Controlnet Mask not implemented yet for clean model"
        # 1. time

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        sample = sample[:, :, :self.num_frames]
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        num_frames = sample.shape[2]
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        emb = emb.repeat_interleave(repeats=num_frames, dim=0)

        if not self.use_image_tokens and encoder_hidden_states.shape[1] > 77:
            encoder_hidden_states = encoder_hidden_states[:, :77]

        if encoder_hidden_states.shape[1] > 77:
            # assert (
            #     encoder_hidden_states.shape[1]-77) % num_frames == 0, f"Encoder shape {encoder_hidden_states.shape}. Num frames = {num_frames}"
            context_text, context_img = encoder_hidden_states[:,
                                                              :77, :], encoder_hidden_states[:, 77:, :]
            context_text = context_text.repeat_interleave(
                repeats=num_frames, dim=0)

            if self.image_encoder_name == "FrozenOpenCLIPImageEmbedder":
                context_img = context_img.repeat_interleave(
                    repeats=num_frames, dim=0)
            else:
                context_img = rearrange(
                    context_img, 'b (t l) c -> (b t) l c', t=num_frames)

            encoder_hidden_states = torch.cat(
                [context_text, context_img], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                repeats=num_frames, dim=0)

        # print(f"ctrl with tokens = {encoder_hidden_states.shape[1]}")
        '''
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            repeats=num_frames, dim=0)
        '''

        # 2. pre-process
        sample = sample.permute(0, 2, 1, 3, 4).reshape(
            (sample.shape[0] * num_frames, -1) + sample.shape[3:])
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(
            controlnet_cond, vq_gan=vq_gan, controlnet_mask=controlnet_mask)

        if num_frames > 1:
            if self.gradient_checkpointing:
                sample = transformer_g_c(
                    self.transformer_in, sample, num_frames)
            else:
                sample = self.transformer_in(
                    sample, num_frames=num_frames, attention_mask=attention_mask).sample

        sample = self.merger(sample * weight_control_sample,
                             weight_control * controlnet_cond)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, num_frames=num_frames)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + \
                (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            # 0.1 to 1.0
            scales = torch.logspace(-1, 0, len(down_block_res_samples) +
                                    1, device=sample.device)

            scales = scales * conditioning_scale
            down_block_res_samples = [
                sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * \
                scales[-1]  # last one
        else:
            down_block_res_samples = [
                sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(
                mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )



def zero_module(module, reset=True):
    if reset:
        for p in module.parameters():
            nn.init.zeros_(p)
    return module
