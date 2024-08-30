
import torch
from models.svd.sgm.modules.diffusionmodules.wrappers import OpenAIWrapper
from einops import rearrange, repeat


class StreamingWrapper(OpenAIWrapper):
    """
    Modelwrapper for StreamingSVD, which holds the CAM model and the base model

    """

    def __init__(self, diffusion_model, controlnet, num_frame_conditioning: int, compile_model: bool = False, pipeline_offloading: bool = False):
        super().__init__(diffusion_model=diffusion_model,
                         compile_model=compile_model)
        self.controlnet = controlnet
        self.num_frame_conditioning = num_frame_conditioning
        self.pipeline_offloading = pipeline_offloading
        if pipeline_offloading:
            raise NotImplementedError(
                "Pipeline offloading for StreamingI2V not implemented yet.")

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs):

        batch_size = kwargs.pop("batch_size")

        # We apply the controlnet model only to the control frames.
        def reduce_to_cond_frames(input):
            input = rearrange(input, "(B F) ... -> B F ...", B=batch_size)
            input = input[:, :self.num_frame_conditioning]
            return rearrange(input, "B F ... -> (B F) ...")

        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        x_ctrl = reduce_to_cond_frames(x)
        t_ctrl = reduce_to_cond_frames(t)

        context = c.get("crossattn", None)
        # controlnet is not using APM so we remove potentially additional tokens
        context_ctrl = context[:, :1]
        context_ctrl = reduce_to_cond_frames(context_ctrl)
        y = c.get("vector", None)
        y_ctrl = reduce_to_cond_frames(y)
        num_video_frames = kwargs.pop("num_video_frames")
        image_only_indicator = kwargs.pop("image_only_indicator")
        ctrl_img_enc_frames = repeat(
            kwargs['ctrl_frames'], "B ... -> (2 B) ... ")
        controlnet_cond = rearrange(
            ctrl_img_enc_frames, "B F ... -> (B F) ...")

        if self.diffusion_model.controlnet_mode:
            hs_control_input, hs_control_mid = self.controlnet(x=x_ctrl,  # video latent
                                                               timesteps=t_ctrl,  # timestep
                                                               context=context_ctrl,  # clip image conditioning
                                                               y=y_ctrl,  # conditionigs, e.g. fps
                                                               controlnet_cond=controlnet_cond,  # control frames
                                                               num_video_frames=self.num_frame_conditioning,
                                                               num_video_frames_conditional=self.num_frame_conditioning,
                                                               image_only_indicator=image_only_indicator[:,
                                                                                                         :self.num_frame_conditioning]
                                                               )
        else:
            hs_control_input = None
            hs_control_mid = None
        kwargs["hs_control_input"] = hs_control_input
        kwargs["hs_control_mid"] = hs_control_mid

        out = self.diffusion_model(
            x=x,
            timesteps=t,
            context=context,  # must be (B F) T C
            y=y,  # must be (B F) 768
            num_video_frames=num_video_frames,
            num_conditional_frames=self.num_frame_conditioning,
            image_only_indicator=image_only_indicator,
            hs_control_input=hs_control_input,
            hs_control_mid=hs_control_mid,
        )
        return out
