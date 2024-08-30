import torch
import torch.nn as nn
from einops import rearrange
from diffusers.models.attention_processor import Attention


class CrossAttention(nn.Module):
    """
    CrossAttention module implements per-pixel temporal attention to fuse the conditional attention module with the base module.

    Args:
        input_channels (int): Number of input channels.
        attention_head_dim (int): Dimension of attention head.
        norm_num_groups (int): Number of groups for GroupNorm normalization (default is 32).

    Attributes:
        attention (Attention): Attention module for computing attention scores.
        norm (torch.nn.GroupNorm): Group normalization layer.
        proj_in (nn.Linear): Linear layer for projecting input data.
        proj_out (nn.Linear): Linear layer for projecting output data.
        dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
        forward(hidden_state, encoder_hidden_states, num_frames, num_conditional_frames):
            Forward pass of the CrossAttention module.

    """

    def __init__(self, input_channels, attention_head_dim, norm_num_groups=32):
        super().__init__()
        self.attention = Attention(
            query_dim=input_channels, cross_attention_dim=input_channels, heads=input_channels//attention_head_dim, dim_head=attention_head_dim, bias=False, upcast_attention=False)
        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=input_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(input_channels, input_channels)
        self.proj_out = nn.Linear(input_channels, input_channels)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, hidden_state, encoder_hidden_states, num_frames, num_conditional_frames):
        """
        The input hidden state is normalized, then projected using a linear layer.
        Multi-head cross attention is computed between the hidden state (latent of noisy video) and encoder hidden states (CLIP image encoder).
        The output is projected using a linear layer.
        We apply dropout to the newly generated frames (without the control frames).

        Args:
            hidden_state (torch.Tensor): Input hidden state tensor.
            encoder_hidden_states (torch.Tensor): Encoder hidden states tensor.
            num_frames (int): Number of frames.
            num_conditional_frames (int): Number of conditional frames.

        Returns:
            output (torch.Tensor): Output tensor after processing with attention mechanism.

        """
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

        residual = self.proj_out(attn)  # (B H W) F C
        hidden_state = rearrange(
            hidden_state, "(B F) ... -> B F ...", F=num_frames)
        hidden_state = torch.cat([hidden_state[:, :num_conditional_frames], self.dropout(
            hidden_state[:, num_conditional_frames:])], dim=1)
        hidden_state = rearrange(hidden_state, "B F ... -> (B F) ... ")

        residual = rearrange(
            residual, "(B H W) F C  -> (B F) C H W", H=h, W=w)
        output = hidden_state + residual
        return output


class ConditionalModel(nn.Module):
    """ 
    ConditionalModel module performs the fusion of the conditional attention module to be base model.

    Args:
        input_channels (int): Number of input channels.
        conditional_model (str): Type of conditional model to use. Currently only "cross_attention" is implemented.
        attention_head_dim (int): Dimension of attention head (default is 64).

    Attributes:
        temporal_transformer (CrossAttention): CrossAttention module for temporal transformation.
        conditional_model (str): Type of conditional model used.

    Methods:
        forward(sample, conditioning, num_frames=None, num_conditional_frames=None):
            Forward pass of the ConditionalModel module.

    """

    def __init__(self, input_channels, conditional_model: str, attention_head_dim=64):
        super().__init__()

        if conditional_model == "cross_attention":
            self.temporal_transformer = CrossAttention(
                input_channels=input_channels, attention_head_dim=attention_head_dim)
        else:
            raise NotImplementedError(
                f"mode {conditional_model} not implemented")

        nn.init.zeros_(self.temporal_transformer.proj_out.weight)
        nn.init.zeros_(self.temporal_transformer.proj_out.bias)
        self.conditional_model = conditional_model

    def forward(self, sample, conditioning, num_frames=None, num_conditional_frames=None):
        """
        Forward pass of the ConditionalModel module.

        Args:
            sample (torch.Tensor): Input sample tensor.
            conditioning (torch.Tensor): Conditioning tensor containing the enconding of the conditional frames.
            num_frames (int): Number of frames in the sample.
            num_conditional_frames (int): Number of conditional frames.

        Returns:
            sample (torch.Tensor): Transformed sample tensor.

        """
        sample = rearrange(sample, "(B F) ... -> B F ...", F=num_frames)
        batch_size = sample.shape[0]
        conditioning = rearrange(
            conditioning, "(B F) ... -> B F ...", B=batch_size)

        assert conditioning.ndim == 5
        assert sample.ndim == 5

        conditioning = rearrange(conditioning, "B F C H W -> (B H W) F C")

        sample = rearrange(sample, "B F C H W -> (B F) C H W")

        sample = self.temporal_transformer(
            sample, encoder_hidden_states=conditioning, num_frames=num_frames, num_conditional_frames=num_conditional_frames)

        return sample


if __name__ == "__main__":
    model = CrossAttention(input_channels=320, attention_head_dim=32)
