import torch
import torch.nn as nn
from typing import List, Optional, Union
from models.svd.sgm.util import default
from models.svd.sgm.modules.video_attention import SpatialVideoTransformer
from models.svd.sgm.modules.diffusionmodules.openaimodel import *
from models.diffusion.video_model import VideoResBlock, VideoUNet
from einops import repeat, rearrange
from models.svd.sgm.modules.diffusionmodules.wrappers import OpenAIWrapper


class Merger(nn.Module):
    """
    Merges the controlnet latents with the conditioning embedding (encoding of control frames).

    """

    def __init__(self, merge_mode: str = "addition", input_channels=0, frame_expansion="last_frame") -> None:
        super().__init__()
        self.merge_mode = merge_mode
        self.frame_expansion = frame_expansion

    def forward(self, x, condition_signal, num_video_frames, num_video_frames_conditional):
        x = rearrange(x, "(B F) C H W -> B F C H W", F=num_video_frames)

        condition_signal = rearrange(
            condition_signal, "(B F) C H W -> B F C H W", B=x.shape[0])

        if x.shape[1] - condition_signal.shape[1] > 0:
            if self.frame_expansion == "last_frame":
                fillup_latent = repeat(
                    condition_signal[:, -1], "B C H W -> B F C H W", F=x.shape[1] - condition_signal.shape[1])
            elif self.frame_expansion == "zero":
                fillup_latent = torch.zeros(
                    (x.shape[0], num_video_frames-num_video_frames_conditional, *x.shape[2:]), device=x.device, dtype=x.dtype)

            if self.frame_expansion != "none":
                condition_signal = torch.cat(
                    [condition_signal, fillup_latent], dim=1)

        if self.merge_mode == "addition":
            out = x + condition_signal
        else:
            raise NotImplementedError(
                f"Merging mode {self.merge_mode} not implemented.")

        out = rearrange(out, "B F C H W -> (B F) C H W")
        return out


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
        downsample: bool = True,
        final_3d_conv: bool = False,
        zero_init: bool = True,
        use_controlnet_mask: bool = False,
        use_normalization: bool = False,
    ):
        super().__init__()

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

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            if use_normalization:
                self.norms.append(nn.LayerNorm((channel_in)))
            self.blocks.append(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=stride))
            if use_normalization:
                self.norms.append(nn.LayerNorm((channel_out)))

        self.conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1]+int(use_controlnet_mask), conditioning_embedding_channels, kernel_size=3, padding=1), reset=zero_init
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        if self.use_normalization:
            for block, norm in zip(self.blocks, self.norms):
                embedding = block(embedding)
                embedding = rearrange(embedding, " ... C W H -> ... W H C")
                embedding = norm(embedding)
                embedding = rearrange(embedding, "... W H C -> ... C W H")
                embedding = F.silu(embedding)
        else:
            for block in self.blocks:
                embedding = block(embedding)
                embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)
        return embedding


class ControlNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Union[List[int], int],
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[Union[int, str]] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        video_kernel_size: Union[int, List[int]] = 3,
        use_linear_in_transformer: bool = False,
        adm_in_channels: Optional[int] = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (
            16, 32, 96, 256),
        condition_encoder: str = "",
        use_controlnet_mask: bool = False,
        downsample_controlnet_cond: bool = True,
        use_image_encoder_normalization: bool = False,
        zero_conv_mode: str = "Identity",
        frame_expansion: str = "none",
        merging_mode: str = "addition",
    ):
        super().__init__()
        assert zero_conv_mode == "Identity", "Zero convolution not implemented"

        assert context_dim is not None

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.transformer_depth = transformer_depth
        self.transformer_depth_middle = transformer_depth_middle
        self.context_dim = context_dim
        self.time_downup = time_downup
        self.time_context_dim = time_context_dim
        self.extra_ff_mix_layer = extra_ff_mix_layer
        self.use_spatial_context = use_spatial_context
        self.merge_strategy = merge_strategy
        self.merge_factor = merge_factor
        self.spatial_transformer_attn_type = spatial_transformer_attn_type
        self.video_kernel_size = video_kernel_size
        self.use_linear_in_transformer = use_linear_in_transformer
        self.adm_in_channels = adm_in_channels
        self.disable_temporal_crossattention = disable_temporal_crossattention
        self.max_ddpm_temb_period = max_ddpm_temb_period

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )

            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disabled_sa=False,
        ):
            return SpatialVideoTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in=extra_ff_mix_layer,
                use_spatial_context=use_spatial_context,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temporal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            return VideoResBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
            )

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.merger = Merger(
            merge_mode=merging_mode, input_channels=model_channels, frame_expansion=frame_expansion)

        conditioning_channels = 3 if downsample_controlnet_cond else 4
        block_out_channels = (320, 640, 1280, 1280)

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            conditioning_channels=conditioning_channels,
            block_out_channels=conditioning_embedding_out_channels,
            downsample=downsample_controlnet_cond,
            final_3d_conv=condition_encoder.endswith("3DConv"),
            use_controlnet_mask=use_controlnet_mask,
            use_normalization=use_image_encoder_normalization,
        )

    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        controlnet_cond: th.Tensor,
        context: Optional[th.Tensor] = None,
        y: Optional[th.Tensor] = None,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        num_video_frames_conditional: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
        hs = []
        t_emb = timestep_embedding(
            timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # TODO restrict y to [:self.num_frames] (conditonal frames)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)

        h = x
        for idx, module in enumerate(self.input_blocks):
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
            if idx == 0:
                h = self.merger(h, controlnet_cond, num_video_frames=num_video_frames,
                                num_video_frames_conditional=num_video_frames_conditional)

            hs.append(h)
        h = self.middle_block(
            h,
            emb,
            context=context,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        )

        # 5. Control net blocks

        down_block_res_samples = hs

        mid_block_res_sample = h

        return (down_block_res_samples, mid_block_res_sample)

    @classmethod
    def from_unet(cls,
                  model: OpenAIWrapper,
                  merging_mode: str = "addition",
                  zero_conv_mode: str = "Identity",
                  frame_expansion: str = "none",
                  downsample_controlnet_cond: bool = True,
                  use_image_encoder_normalization: bool = False,
                  use_controlnet_mask: bool = False,
                  condition_encoder: str = "",
                  conditioning_embedding_out_channels: List[int] = None,

                  ):

        unet: VideoUNet = model.diffusion_model

        controlnet = cls(in_channels=unet.in_channels,
                         model_channels=unet.model_channels,
                         out_channels=unet.out_channels,
                         num_res_blocks=unet.num_res_blocks,
                         attention_resolutions=unet.attention_resolutions,
                         dropout=unet.dropout,
                         channel_mult=unet.channel_mult,
                         conv_resample=unet.conv_resample,
                         dims=unet.dims,
                         num_classes=unet.num_classes,
                         use_checkpoint=unet.use_checkpoint,
                         num_heads=unet.num_heads,
                         num_head_channels=unet.num_head_channels,
                         num_heads_upsample=unet.num_heads_upsample,
                         use_scale_shift_norm=unet.use_scale_shift_norm,
                         resblock_updown=unet.resblock_updown,
                         transformer_depth=unet.transformer_depth,
                         transformer_depth_middle=unet.transformer_depth_middle,
                         context_dim=unet.context_dim,
                         time_downup=unet.time_downup,
                         time_context_dim=unet.time_context_dim,
                         extra_ff_mix_layer=unet.extra_ff_mix_layer,
                         use_spatial_context=unet.use_spatial_context,
                         merge_strategy=unet.merge_strategy,
                         merge_factor=unet.merge_factor,
                         spatial_transformer_attn_type=unet.spatial_transformer_attn_type,
                         video_kernel_size=unet.video_kernel_size,
                         use_linear_in_transformer=unet.use_linear_in_transformer,
                         adm_in_channels=unet.adm_in_channels,
                         disable_temporal_crossattention=unet.disable_temporal_crossattention,
                         max_ddpm_temb_period=unet.max_ddpm_temb_period,  # up to here unet params
                         merging_mode=merging_mode,
                         zero_conv_mode=zero_conv_mode,
                         frame_expansion=frame_expansion,
                         downsample_controlnet_cond=downsample_controlnet_cond,
                         use_image_encoder_normalization=use_image_encoder_normalization,
                         use_controlnet_mask=use_controlnet_mask,
                         condition_encoder=condition_encoder,
                         conditioning_embedding_out_channels=conditioning_embedding_out_channels,
                         )
        controlnet: ControlNet

        return controlnet


def zero_module(module, reset=True):
    if reset:
        for p in module.parameters():
            nn.init.zeros_(p)
    return module
