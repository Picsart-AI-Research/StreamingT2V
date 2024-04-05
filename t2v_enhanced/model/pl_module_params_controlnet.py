from typing import Union, Any, Dict, List, Optional, Callable
from t2v_enhanced.model import pl_module_extension
from t2v_enhanced.model.diffusers_conditional.models.controlnet.image_embedder import AbstractEncoder
from t2v_enhanced.model.requires_grad_setter import LayerConfig as LayerConfigNew
from t2v_enhanced.model import video_noise_generator


def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    return cls


class LayerConfig():
    def __init__(self,
                 update_with_full_lr: Optional[Union[List[str],
                                                     List[List[str]]]] = None,
                 exclude: Optional[List[str]] = None,
                 deactivate_all_grads: bool = True,
                 ) -> None:
        self.deactivate_all_grads = deactivate_all_grads
        if exclude is not None:
            self.exclude = exclude
        if update_with_full_lr is not None:
            self.update_with_full_lr = update_with_full_lr

    def __str__(self) -> str:
        str = f"Deactivate all gradients first={self.deactivate_all_grads}. "
        if hasattr(self, "update_with_full_lr"):
            str += f"Then activating gradients for: {self.update_with_full_lr}. "
        if hasattr(self, "exclude"):
            str += f"Finally, excluding: {self.exclude}. "
        return str


class OptimizerParams():
    def __init__(self,
                 learning_rate: float,
                 # Default value due to legacy
                 layers_config: Union[LayerConfig, LayerConfigNew] = None,
                 layers_config_base: LayerConfig = None,  # Default value due to legacy
                 use_warmup: bool = False,
                 warmup_steps: int = 10000,
                 warmup_start_factor: float = 1e-5,
                 learning_rate_spatial: float = 0.0,
                 use_8_bit_adam: bool = False,
                 noise_generator: Union[pl_module_extension.NoiseGenerator,
                                        video_noise_generator.NoiseGenerator] = None,
                 noise_decomposition: pl_module_extension.NoiseDecomposition = None,
                 perceptual_loss: bool = False,
                 noise_offset: float = 0.0,
                 split_opt_by_node: bool = False,
                 reset_prediction_type_to_eps: bool = False,
                 train_val_sampler_may_differ: bool = False,
                 measure_similarity: bool = False,
                 similarity_loss: bool = False,
                 similarity_loss_weight: float = 1.0,
                 loss_conditional_weight: float = 0.0,
                 loss_conditional_weight_convex: bool = False,
                 loss_conditional_change_after_step: int = 0,
                 mask_conditional_frames: bool = False,
                 sample_from_noise: bool = True,
                 mask_alternating: bool = False,
                 uncondition_freq: int = -1,
                 no_text_condition_control: bool = False,
                 inject_image_into_input: bool = False,
                 inject_at_T: bool = False,
                 resampling_steps: int = 1,
                 control_freq_in_resample: int = 1,
                 resample_to_T: bool = False,
                 adaptive_loss_reweight: bool = False,
                 load_resampler_from_ckpt: str = "",
                 skip_controlnet_branch: bool = False,
                 use_fps_conditioning: bool = False,
                 num_frame_embeddings_range: int = 16,
                 start_frame_training: int = 0,
                 start_frame_ctrl: int = 0,
                 load_trained_base_model_and_resampler_from_ckpt: str = "",
                 load_trained_controlnet_from_ckpt: str = "",
                 # fill_up_frame_to_video: bool = False,
                 ) -> None:
        self.use_warmup = use_warmup
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.learning_rate_spatial = learning_rate_spatial
        self.learning_rate = learning_rate
        self.use_8_bit_adam = use_8_bit_adam
        self.layers_config = layers_config
        self.noise_generator = noise_generator
        self.perceptual_loss = perceptual_loss
        self.noise_decomposition = noise_decomposition
        self.noise_offset = noise_offset
        self.split_opt_by_node = split_opt_by_node
        self.reset_prediction_type_to_eps = reset_prediction_type_to_eps
        self.train_val_sampler_may_differ = train_val_sampler_may_differ
        self.measure_similarity = measure_similarity
        self.similarity_loss = similarity_loss
        self.similarity_loss_weight = similarity_loss_weight
        self.loss_conditional_weight = loss_conditional_weight
        self.loss_conditional_change_after_step = loss_conditional_change_after_step
        self.mask_conditional_frames = mask_conditional_frames
        self.loss_conditional_weight_convex = loss_conditional_weight_convex
        self.sample_from_noise = sample_from_noise
        self.layers_config_base = layers_config_base
        self.mask_alternating = mask_alternating
        self.uncondition_freq = uncondition_freq
        self.no_text_condition_control = no_text_condition_control
        self.inject_image_into_input = inject_image_into_input
        self.inject_at_T = inject_at_T
        self.resampling_steps = resampling_steps
        self.control_freq_in_resample = control_freq_in_resample
        self.resample_to_T = resample_to_T
        self.adaptive_loss_reweight = adaptive_loss_reweight
        self.load_resampler_from_ckpt = load_resampler_from_ckpt
        self.skip_controlnet_branch = skip_controlnet_branch
        self.use_fps_conditioning = use_fps_conditioning
        self.num_frame_embeddings_range = num_frame_embeddings_range
        self.start_frame_training = start_frame_training
        self.load_trained_base_model_and_resampler_from_ckpt = load_trained_base_model_and_resampler_from_ckpt
        self.load_trained_controlnet_from_ckpt = load_trained_controlnet_from_ckpt
        self.start_frame_ctrl = start_frame_ctrl
        if start_frame_ctrl < 0:
            print("new format start frame cannot be negative")
            exit()

        # self.fill_up_frame_to_video = fill_up_frame_to_video

    @property
    def learning_rate_spatial(self):
        return self._learning_rate_spatial

    # legacy code that maps the state None or '-1' to '0.0'
    # so 0.0 indicated no spatial learning rate is selected
    @learning_rate_spatial.setter
    def learning_rate_spatial(self, value):
        if value is None or value == -1:
            value = 0
        self._learning_rate_spatial = value


# Legacy class
class SchedulerParams():
    def __init__(self,
                 use_warmup: bool = False,
                 warmup_steps: int = 10000,
                 warmup_start_factor: float = 1e-5,
                 ) -> None:
        self.use_warmup = use_warmup
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor



class CrossFrameAttentionParams():

    def __init__(self, attent_on: List[int], masking=False) -> None:
        self.attent_on = attent_on
        self.masking = masking


class InferenceParams():
    def __init__(self,
                 width: int,
                 height: int,
                 video_length: int,
                 guidance_scale: float = 7.5,
                 use_dec_scaling: bool = True,
                 frame_rate: int = 2,
                 num_inference_steps: int = 50,
                 eta: float = 0.0,
                 n_autoregressive_generations: int = 1,
                 mode: str = "long_video",
                 start_from_real_input: bool = True,
                 eval_loss_metrics: bool = False,
                 scheduler_cls: str = "",
                 negative_prompt: str = "",
                 conditioning_from_all_past: bool = False,
                 validation_samples: int = 80,
                 conditioning_type: str = "last_chunk",
                 result_formats: List[str] = ["eval_gif", "gif", "mp4"],
                 concat_video: bool = True,
                 seed: int = 33,
                 ):
        self.width = width
        self.height = height
        self.video_length = video_length if isinstance(
            video_length, int) else int(video_length)
        self.guidance_scale = guidance_scale
        self.use_dec_scaling = use_dec_scaling
        self.frame_rate = frame_rate
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.negative_prompt = negative_prompt
        self.n_autoregressive_generations = n_autoregressive_generations
        self.mode = mode
        self.start_from_real_input = start_from_real_input
        self.eval_loss_metrics = eval_loss_metrics
        self.scheduler_cls = scheduler_cls
        self.conditioning_from_all_past = conditioning_from_all_past
        self.validation_samples = validation_samples
        self.conditioning_type = conditioning_type
        self.result_formats = result_formats
        self.concat_video = concat_video
        self.seed = seed

    def to_dict(self):

        keys = [entry for entry in dir(self) if not callable(getattr(
            self, entry)) and not entry.startswith("__")]

        result_dict = {}
        for key in keys:
            result_dict[key] = getattr(self, key)
        return result_dict


@auto_str
class AttentionMaskParams():

    def __init__(self,
                 temporal_self_attention_only_on_conditioning: bool = False,
                 temporal_self_attention_mask_included_itself: bool = False,
                 spatial_attend_on_condition_frames: bool = False,
                 temp_attend_on_neighborhood_of_condition_frames: bool = False,
                 temp_attend_on_uncond_include_past: bool = False,
                 ) -> None:
        self.temporal_self_attention_mask_included_itself = temporal_self_attention_mask_included_itself
        self.spatial_attend_on_condition_frames = spatial_attend_on_condition_frames
        self.temp_attend_on_neighborhood_of_condition_frames = temp_attend_on_neighborhood_of_condition_frames
        self.temporal_self_attention_only_on_conditioning = temporal_self_attention_only_on_conditioning
        self.temp_attend_on_uncond_include_past = temp_attend_on_uncond_include_past

        assert not temp_attend_on_neighborhood_of_condition_frames or not temporal_self_attention_only_on_conditioning


class UNetParams():

    def __init__(self,
                 conditioning_embedding_out_channels: List[int],
                 ckpt_spatial_layers: str = "",
                 pipeline_repo: str = "",
                 unet_from_diffusers: bool = True,
                 spatial_latent_input: bool = False,
                 num_frame_conditioning: int = 1,
                 pipeline_class: str = "t2v_enhanced.model.model.controlnet.pipeline_text_to_video_w_controlnet_synth.TextToVideoSDPipeline",
                 frame_expansion: str = "last_frame",
                 downsample_controlnet_cond: bool = True,
                 num_frames: int = 1,
                 pre_transformer_in_cond: bool = False,
                 num_tranformers: int = 1,
                 zero_conv_3d: bool = False,
                 merging_mode: str = "addition",
                 compute_only_conditioned_frames: bool = False,
                 condition_encoder: str = "",
                 zero_conv_mode: str = "2d",
                 clean_model: bool = False,
                 merging_mode_base: str = "addition",
                 attention_mask_params: AttentionMaskParams = None,
                 attention_mask_params_base: AttentionMaskParams = None,
                 modelscope_input_format: bool = True,
                 temporal_self_attention_only_on_conditioning: bool = False,
                 temporal_self_attention_mask_included_itself: bool = False,
                 use_post_merger_zero_conv: bool = False,
                 weight_control_sample: float = 1.0,
                 use_controlnet_mask: bool = False,
                 random_mask_shift: bool = False,
                 random_mask: bool = False,
                 use_resampler: bool = False,
                 unet_from_pipe: bool = False,
                 unet_operates_on_2d: bool = False,
                 image_encoder: str = "CLIP",
                 use_standard_attention_processor: bool = True,
                 num_frames_before_chunk: int = 0,
                 resampler_type: str = "single_frame",
                 resampler_cls: str = "",
                 resampler_merging_layers: int = 1,
                 image_encoder_obj: AbstractEncoder = None,
                 cfg_text_image: bool = False,
                 aggregation: str = "last_out",
                 resampler_random_shift: bool = False,
                 img_cond_alpha_per_frame: bool = False,
                 num_control_input_frames: int = -1,
                 use_image_encoder_normalization: bool = False,
                 use_of: bool = False,
                 ema_param: float = -1.0,
                 concat: bool = False,
                 use_image_tokens_main: bool = True,
                 use_image_tokens_ctrl: bool = False,
                 ):

        self.ckpt_spatial_layers = ckpt_spatial_layers
        self.pipeline_repo = pipeline_repo
        self.unet_from_diffusers = unet_from_diffusers
        self.spatial_latent_input = spatial_latent_input
        self.pipeline_class = pipeline_class
        self.num_frame_conditioning = num_frame_conditioning
        if num_control_input_frames == -1:
            self.num_control_input_frames = num_frame_conditioning
        else:
            self.num_control_input_frames = num_control_input_frames

        self.conditioning_embedding_out_channels = conditioning_embedding_out_channels
        self.frame_expansion = frame_expansion
        self.downsample_controlnet_cond = downsample_controlnet_cond
        self.num_frames = num_frames
        self.pre_transformer_in_cond = pre_transformer_in_cond
        self.num_tranformers = num_tranformers
        self.zero_conv_3d = zero_conv_3d
        self.merging_mode = merging_mode
        self.compute_only_conditioned_frames = compute_only_conditioned_frames
        self.clean_model = clean_model
        self.condition_encoder = condition_encoder
        self.zero_conv_mode = zero_conv_mode
        self.merging_mode_base = merging_mode_base
        self.modelscope_input_format = modelscope_input_format
        assert not temporal_self_attention_only_on_conditioning, "This parameter is only here for backward compatibility. Set AttentionMaskParams instead."
        assert not temporal_self_attention_mask_included_itself, "This parameter is only here for backward compatibility. Set AttentionMaskParams instead."
        if attention_mask_params is not None and attention_mask_params_base is None:
            attention_mask_params_base = attention_mask_params
        if attention_mask_params is None:
            attention_mask_params = AttentionMaskParams()
        if attention_mask_params_base is None:
            attention_mask_params_base = AttentionMaskParams()
        self.attention_mask_params = attention_mask_params
        self.attention_mask_params_base = attention_mask_params_base
        self.weight_control_sample = weight_control_sample
        self.use_controlnet_mask = use_controlnet_mask
        self.random_mask_shift = random_mask_shift
        self.random_mask = random_mask
        self.use_resampler = use_resampler
        self.unet_from_pipe = unet_from_pipe
        self.unet_operates_on_2d = unet_operates_on_2d
        self.image_encoder = image_encoder_obj
        self.use_standard_attention_processor = use_standard_attention_processor
        self.num_frames_before_chunk = num_frames_before_chunk
        self.resampler_type = resampler_type
        self.resampler_cls = resampler_cls
        self.resampler_merging_layers = resampler_merging_layers
        self.cfg_text_image = cfg_text_image
        self.aggregation = aggregation
        self.resampler_random_shift = resampler_random_shift
        self.img_cond_alpha_per_frame = img_cond_alpha_per_frame
        self.use_image_encoder_normalization = use_image_encoder_normalization
        self.use_of = use_of
        self.ema_param = ema_param
        self.concat = concat
        self.use_image_tokens_main = use_image_tokens_main
        self.use_image_tokens_ctrl = use_image_tokens_ctrl
        assert not use_post_merger_zero_conv

        if spatial_latent_input:
            assert unet_from_diffusers, "Spatial latent input only implemented by original diffusers model. Set 'model.unet_params.unet_from_diffusers=True'."
