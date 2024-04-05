from t2v_enhanced.model.pl_module_params_controlnet import AttentionMaskParams
import torch


class MaskGenerator():

    def __init__(self, params: AttentionMaskParams, num_frame_conditioning, num_frames):
        self.params = params
        self.num_frame_conditioning = num_frame_conditioning
        self.num_frames = num_frames
    def get_mask(self, precision, device):

        params = self.params
        if params.temporal_self_attention_only_on_conditioning:
            with torch.no_grad():
                attention_mask = torch.zeros((1, self.num_frames, self.num_frames), dtype=torch.float16 if precision.startswith(
                    "16") else torch.float32, device=device)
                for frame in range(self.num_frame_conditioning, self.num_frames):
                    attention_mask[:, frame,
                                   self.num_frame_conditioning:] = float("-inf")
                    if params.temporal_self_attention_mask_included_itself:
                        attention_mask[:, frame, frame] = 0
                    if params.temp_attend_on_uncond_include_past:
                        attention_mask[:, frame, :frame] = 0
        else:
            attention_mask = None
        return attention_mask
