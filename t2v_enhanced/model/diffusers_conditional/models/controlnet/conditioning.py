import diffusers
from diffusers.models.transformer_temporal import TransformerTemporalModel, TransformerTemporalModelOutput
import torch.nn as nn
from einops import rearrange
from diffusers.models.attention_processor import Attention
# from t2v_enhanced.model.diffusers_conditional.models.controlnet.attention_processor import Attention
from t2v_enhanced.model.diffusers_conditional.models.controlnet.transformer_temporal_crossattention import TransformerTemporalModel as TransformerTemporalModelCrossAttn
import torch


class CrossAttention(nn.Module):

    def __init__(self, input_channels, attention_head_dim, norm_num_groups=32):
        super().__init__()
        self.attention = Attention(
            query_dim=input_channels, cross_attention_dim=input_channels, heads=input_channels//attention_head_dim, dim_head=attention_head_dim, bias=False, upcast_attention=False)
        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=input_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(input_channels, input_channels)
        self.proj_out = nn.Linear(input_channels, input_channels)

    def forward(self, hidden_state, encoder_hidden_states, num_frames):
        h, w = hidden_state.shape[2], hidden_state.shape[3]
        hidden_state_norm = rearrange(
            hidden_state, "(B F) C H W -> B C F H W", F=num_frames)
        hidden_state_norm = self.norm(hidden_state_norm)
        hidden_state_norm = rearrange(
            hidden_state_norm, "B C F H W -> (B H W) F C")
        hidden_state_norm = self.proj_in(hidden_state_norm)
        attn = self.attention(hidden_state_norm,
                              encoder_hidden_states=encoder_hidden_states,
                              attention_mask=None,
                              )
        # proj_out

        residual = self.proj_out(attn)

        residual = rearrange(
            residual, "(B H W) F C  -> (B F) C H W", H=h, W=w)
        output = hidden_state + residual
        return TransformerTemporalModelOutput(sample=output)


class ConditionalModel(nn.Module):

    def __init__(self, input_channels, conditional_model: str, attention_head_dim=64):
        super().__init__()
        num_layers = 1
        if "_layers_" in conditional_model:
            config = conditional_model.split("_layers_")
            conditional_model = config[0]
            num_layers = int(config[1])

        if conditional_model == "self_cross_transformer":
            self.temporal_transformer = TransformerTemporalModel(num_attention_heads=input_channels//attention_head_dim, attention_head_dim=attention_head_dim, in_channels=input_channels,
                                                                 double_self_attention=False, cross_attention_dim=input_channels)
        elif conditional_model == "cross_transformer":
            self.temporal_transformer = TransformerTemporalModelCrossAttn(num_attention_heads=input_channels//attention_head_dim, attention_head_dim=attention_head_dim, in_channels=input_channels,
                                                                          double_self_attention=False, cross_attention_dim=input_channels, num_layers=num_layers)
        elif conditional_model == "cross_attention":
            self.temporal_transformer = CrossAttention(
                input_channels=input_channels, attention_head_dim=attention_head_dim)
        elif conditional_model == "test_conv":
            self.temporal_transformer = nn.Conv2d(
                input_channels, input_channels, kernel_size=1)
        else:
            raise NotImplementedError(
                f"mode {conditional_model} not implemented")
        if conditional_model != "test_conv":
            nn.init.zeros_(self.temporal_transformer.proj_out.weight)
            nn.init.zeros_(self.temporal_transformer.proj_out.bias)
        else:
            nn.init.zeros_(self.temporal_transformer.weight)
            nn.init.zeros_(self.temporal_transformer.bias)
        self.conditional_model = conditional_model

    def forward(self, sample, conditioning, num_frames=None):

        assert conditioning.ndim == 5
        assert sample.ndim == 5
        if self.conditional_model != "test_conv":
            conditioning = rearrange(conditioning, "B F C H W -> (B H W) F C")

            num_frames = sample.shape[1]

            sample = rearrange(sample, "B F C H W -> (B F) C H W")

            sample = self.temporal_transformer(
                sample, encoder_hidden_states=conditioning, num_frames=num_frames).sample

            sample = rearrange(
                sample, "(B F) C H W -> B F C H W", F=num_frames)
        else:

            conditioning = rearrange(conditioning, "B F C H W -> (B F) C H W")
            f = sample.shape[1]
            sample = rearrange(sample, "B F C H W -> (B F) C H W")
            sample = sample + self.temporal_transformer(conditioning)
            sample = rearrange(sample, "(B F) C H W -> B F C H W", F=f)
        return sample
