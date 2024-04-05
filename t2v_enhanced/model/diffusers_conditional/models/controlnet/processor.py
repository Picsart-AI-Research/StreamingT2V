from einops import repeat, rearrange
from typing import Callable, Optional, Union
from t2v_enhanced.model.diffusers_conditional.models.controlnet.attention_processor import Attention
# from t2v_enhanced.model.diffusers_conditional.controldiffusers.models.attention import Attention
from diffusers.utils.import_utils import is_xformers_available
from t2v_enhanced.model.pl_module_params_controlnet import AttentionMaskParams
import torch
import torch.nn.functional as F
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def set_use_memory_efficient_attention_xformers(
    model, num_frame_conditioning: int, num_frames: int, attention_mask_params: AttentionMaskParams, valid: bool = True, attention_op: Optional[Callable] = None
) -> None:
    # Recursively walk through all the children.
    # Any children which exposes the set_use_memory_efficient_attention_xformers method
    # gets the message
    def fn_recursive_set_mem_eff(module: torch.nn.Module):
        if hasattr(module, "set_processor"):

            module.set_processor(XFormersAttnProcessor(attention_op=attention_op,
                                                       num_frame_conditioning=num_frame_conditioning,
                                                       num_frames=num_frames,
                                                       attention_mask_params=attention_mask_params,)
                                 )

        for child in module.children():
            fn_recursive_set_mem_eff(child)

    for module in model.children():
        if isinstance(module, torch.nn.Module):
            fn_recursive_set_mem_eff(module)


class XFormersAttnProcessor:
    def __init__(self,
                 attention_mask_params: AttentionMaskParams,
                 attention_op: Optional[Callable] = None,
                 num_frame_conditioning: int = None,
                 num_frames: int = None,
                 use_image_embedding: bool = False,
                 ):
        self.attention_op = attention_op
        self.num_frame_conditioning = num_frame_conditioning
        self.num_frames = num_frames
        self.temp_attend_on_neighborhood_of_condition_frames = attention_mask_params.temp_attend_on_neighborhood_of_condition_frames
        self.spatial_attend_on_condition_frames = attention_mask_params.spatial_attend_on_condition_frames
        self.use_image_embedding = use_image_embedding

    def __call__(self, attn: Attention, hidden_states, hidden_state_height=None, hidden_state_width=None, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        key_img = None
        value_img = None
        hidden_states_img = None
        if attention_mask is not None:
            attention_mask = repeat(
                attention_mask, "1 F D -> B F D", B=batch_size)

        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)
        default_attention = not hasattr(attn, "is_spatial_attention")
        if default_attention:
            assert not self.temp_attend_on_neighborhood_of_condition_frames, "special attention must be implemented with new interface"
            assert not self.spatial_attend_on_condition_frames, "special attention must be implemented with new interface"
        is_spatial_attention = attn.is_spatial_attention if hasattr(
            attn, "is_spatial_attention") else False
        use_image_embedding = attn.use_image_embedding if hasattr(
            attn, "use_image_embedding") else False

        if is_spatial_attention and use_image_embedding and attn.cross_attention_mode:
            assert not self.spatial_attend_on_condition_frames, "Not implemented together with image embedding"

            alpha = attn.alpha
            encoder_hidden_states_txt = encoder_hidden_states[:, :77, :]

            encoder_hidden_states_mixed = attn.conv(encoder_hidden_states)
            encoder_hidden_states_mixed = attn.conv_ln(encoder_hidden_states_mixed)
            encoder_hidden_states = encoder_hidden_states_txt + encoder_hidden_states_mixed * F.silu(alpha)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)




        if not default_attention and not is_spatial_attention and self.temp_attend_on_neighborhood_of_condition_frames and not attn.cross_attention_mode:
            # normal attention
            query_condition = query[:, :self.num_frame_conditioning]
            query_condition = attn.head_to_batch_dim(
                query_condition).contiguous()
            key_condition = key
            value_condition = value
            key_condition = attn.head_to_batch_dim(key_condition).contiguous()
            value_condition = attn.head_to_batch_dim(
                value_condition).contiguous()
            hidden_states_condition = xformers.ops.memory_efficient_attention(
                query_condition, key_condition, value_condition, attn_bias=None, op=self.attention_op, scale=attn.scale
            )
            hidden_states_condition = hidden_states_condition.to(query.dtype)
            hidden_states_condition = attn.batch_to_head_dim(
                hidden_states_condition)
            #
            query_uncondition = query[:, self.num_frame_conditioning:]

            key = key[:, :self.num_frame_conditioning]
            value = value[:, :self.num_frame_conditioning]
            key = rearrange(key, "(B W H) F C -> B W H F C",
                            H=hidden_state_height, W=hidden_state_width)
            value = rearrange(value, "(B W H) F C -> B W H F C",
                              H=hidden_state_height, W=hidden_state_width)

            keys = []
            values = []
            for shifts_width in [-1, 0, 1]:
                for shifts_height in [-1, 0, 1]:
                    keys.append(torch.roll(key, shifts=(
                        shifts_width, shifts_height), dims=(1, 2)))
                    values.append(torch.roll(value, shifts=(
                        shifts_width, shifts_height), dims=(1, 2)))
            key = rearrange(torch.cat(keys, dim=3), "B W H F C -> (B W H) F C")
            value = rearrange(torch.cat(values, dim=3),
                              'B W H F C -> (B W H) F C')

            query = attn.head_to_batch_dim(query_uncondition).contiguous()
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()

            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
            )
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = torch.cat(
                [hidden_states_condition, hidden_states], dim=1)
        elif not default_attention and is_spatial_attention and self.spatial_attend_on_condition_frames and not attn.cross_attention_mode:
            # (B F) W H C -> B F W H C
            query_condition = rearrange(
                query, "(B F) S C -> B F S C", F=self.num_frames)
            query_condition = query_condition[:, :self.num_frame_conditioning]
            query_condition = rearrange(
                query_condition, "B F S C -> (B F) S C")
            query_condition = attn.head_to_batch_dim(
                query_condition).contiguous()

            key_condition = rearrange(
                key, "(B F) S C -> B F S C", F=self.num_frames)
            key_condition = key_condition[:, :self.num_frame_conditioning]
            key_condition = rearrange(key_condition, "B F S C -> (B F) S C")

            value_condition = rearrange(
                value, "(B F) S C -> B F S C", F=self.num_frames)
            value_condition = value_condition[:, :self.num_frame_conditioning]
            value_condition = rearrange(
                value_condition, "B F S C -> (B F) S C")

            key_condition = attn.head_to_batch_dim(key_condition).contiguous()
            value_condition = attn.head_to_batch_dim(
                value_condition).contiguous()
            hidden_states_condition = xformers.ops.memory_efficient_attention(
                query_condition, key_condition, value_condition, attn_bias=None, op=self.attention_op, scale=attn.scale
            )
            hidden_states_condition = hidden_states_condition.to(query.dtype)
            hidden_states_condition = attn.batch_to_head_dim(
                hidden_states_condition)

            query_uncondition = rearrange(
                query, "(B F) S C -> B F S C", F=self.num_frames)
            query_uncondition = query_uncondition[:,
                                                  self.num_frame_conditioning:]
            key_uncondition = rearrange(
                key, "(B F) S C -> B F S C", F=self.num_frames)
            value_uncondition = rearrange(
                value, "(B F) S C -> B F S C", F=self.num_frames)
            key_uncondition = key_uncondition[:,
                                              self.num_frame_conditioning-1, None]
            value_uncondition = value_uncondition[:,
                                                  self.num_frame_conditioning-1, None]
            # if self.trainer.training:
            # import pdb
            # pdb.set_trace()
            # print("now")
            query_uncondition = rearrange(
                query_uncondition, "B F S C -> (B F) S C")
            key_uncondition = repeat(rearrange(
                key_uncondition, "B F S C -> B (F S) C"), "B T C -> (B F) T C", F=self.num_frames-self.num_frame_conditioning)
            value_uncondition = repeat(rearrange(
                value_uncondition, "B F S C -> B (F S) C"), "B T C -> (B F) T C", F=self.num_frames-self.num_frame_conditioning)
            query_uncondition = attn.head_to_batch_dim(
                query_uncondition).contiguous()
            key_uncondition = attn.head_to_batch_dim(
                key_uncondition).contiguous()
            value_uncondition = attn.head_to_batch_dim(
                value_uncondition).contiguous()
            hidden_states_uncondition = xformers.ops.memory_efficient_attention(
                query_uncondition, key_uncondition, value_uncondition, attn_bias=None, op=self.attention_op, scale=attn.scale
            )
            hidden_states_uncondition = hidden_states_uncondition.to(
                query.dtype)
            hidden_states_uncondition = attn.batch_to_head_dim(
                hidden_states_uncondition)
            hidden_states = torch.cat([rearrange(hidden_states_condition, "(B F) S C -> B F S C", F=self.num_frame_conditioning), rearrange(
                hidden_states_uncondition, "(B F) S C -> B F S C", F=self.num_frames-self.num_frame_conditioning)], dim=1)
            hidden_states = rearrange(hidden_states, "B F S C -> (B F) S C")
        else:
            query = attn.head_to_batch_dim(query).contiguous()
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()

            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
            )

            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
