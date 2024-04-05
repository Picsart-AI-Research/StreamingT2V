import torch
import torch.nn as nn
from typing import Union
from torch.nn.common_types import _size_2_t


class Conv2D_SubChannels(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        if prefix+"weight" in state_dict and ((state_dict[prefix+"weight"].shape[0] > self.out_channels) or (state_dict[prefix+"weight"].shape[1] > self.in_channels)):
            print(
                f"Model checkpoint has too many channels. Excluding channels of convolution {prefix}.")
            if self.bias is not None:
                bias = state_dict[prefix+"bias"][:self.out_channels]
                state_dict[prefix+"bias"] = bias
                del bias

            weight = state_dict[prefix+"weight"]
            state_dict[prefix+"weight"] = weight[:self.out_channels,
                                                 :self.in_channels]
            del weight

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class Conv2D_ExtendedChannels(nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 in_channel_extension: int = 0,
                 out_channel_extension: int = 0,
                 ) -> None:
        super().__init__(in_channels+in_channel_extension, out_channels+out_channel_extension, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        print(f"Call extend channel loader with {prefix}")
        if prefix+"weight" in state_dict and (state_dict[prefix+"weight"].shape[0] < self.out_channels or state_dict[prefix+"weight"].shape[1] < self.in_channels):
            print(
                f"Model checkpoint has insufficient channels. Extending channels of convolution {prefix} by adding zeros.")
            if self.bias is not None:
                bias = state_dict[prefix+"bias"]
                state_dict[prefix+"bias"] = torch.cat(
                    [bias, torch.zeros(self.out_channels-len(bias), dtype=bias.dtype, layout=bias.layout, device=bias.device)])
                del bias

            weight = state_dict[prefix+"weight"]
            extended_weight = torch.zeros(self.out_channels, self.in_channels,
                                          weight.shape[2], weight.shape[3], device=weight.device, dtype=weight.dtype, layout=weight.layout)
            extended_weight[:weight.shape[0], :weight.shape[1]] = weight
            state_dict[prefix+"weight"] = extended_weight
            del extended_weight
            del weight

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


if __name__ == "__main__":
    class MyModel(nn.Module):

        def __init__(self, conv_type: str, c_in, c_out, in_extension, out_extension) -> None:
            super().__init__()

            if not conv_type == "normal":

                self.conv1 = Conv2D_ExtendedChannels(
                    c_in, c_out, 3, padding=1, in_channel_extension=in_extension, out_channel_extension=out_extension, bias=True)

            else:
                self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=True)

        def forward(self, x):
            return self.conv1(x)

    c_in = 9
    c_out = 12
    c_in_ext = 0
    c_out_ext = 3
    model = MyModel("normal", c_in, c_out, c_in_ext, c_out_ext)

    input = torch.randn((4, c_in+c_in_ext, 128, 128))
    out_normal = model(input[:, :c_in])
    torch.save(model.state_dict(), "model_dummy.py")

    model_2 = MyModel("special", c_in, c_out, c_in_ext, c_out_ext)
    model_2.load_state_dict(torch.load("model_dummy.py"))
    out_model_2 = model_2(input)
    out_special = out_model_2[:, :c_out]

    out_new = out_model_2[:, c_out:]
    model_3 = MyModel("special", c_in, c_out, c_in_ext, c_out_ext)
    model_3.load_state_dict(model_2.state_dict())
    # out_model_2 = model_2(input)
    # out_special = out_model_2[:, :c_out]

    print(
        f"Difference: Forward pass with extended convolution minus initial convolution: {(out_normal-out_special).abs().max()}")

    print(f"Compared tensors with shape: ",
          out_normal.shape, out_special.shape)

    if model_3.conv1.bias is not None:
        criterion = nn.MSELoss()

        before_opt = model_3.conv1.bias.detach().clone()
        target = torch.ones_like(out_model_2)
        optimizer = torch.optim.SGD(
            model_3.parameters(), lr=0.01, momentum=0.9)
        for iter in range(10):
            optimizer.zero_grad()
            out = model_3(input)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        print(
            f"Weights before and after are the same? {before_opt[c_out:].detach()} | {model_3.conv1.bias[c_out:].detach()} ")
        print(model_3.conv1.bias, model_2.conv1.bias)
