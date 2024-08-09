import pickle
from typing import Tuple, Sequence
import torch.nn as nn
from torch import Tensor

from einops import rearrange

"""Conv modules"""


def Conv2d(*args, **kwargs) -> nn.Module:
    return nn.Conv2d(*args, **kwargs)


def ConvTranspose2d(*args, **kwargs) -> nn.Module:
    return nn.ConvTranspose2d(*args, **kwargs)


def Downsample2d(
    in_channels: int,
    out_channels: int,
    factor: Tuple[int, int],
    kernel_multiplier: int = 2,
) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[fac * kernel_multiplier + 1 for fac in factor],
        stride=factor,
        padding=[fac * (kernel_multiplier // 2) for fac in factor],
    )


def Upsample2d(
    in_channels: int,
    out_channels: int,
    factor: Tuple[int, int],
    pads: Tuple[int, int],
    kernel_multiplier: int = 2,
) -> nn.Module:
    # if factor == [1, 1]:
    #     return ConvTranspose2d(
    #         in_channels=in_channels,
    #         out_channels=out_channels,
    #         kernel_size=(3, 3),
    #         # padding=pads,
    #         # output_padding=pads,
    #     )

    padding = [fac * (kernel_multiplier // 2) for fac in factor]
    # output_padding = [fact % 2 for fact in factor]
    # output_padding = [0, 0]
    foo = 1

    return ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor,
        stride=factor,
        # padding=padding,
    )


class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = [3, 3],
        stride: Tuple[int, int] = [1, 1],
        padding: Tuple[int, int] = [1, 1],
        dilation: Tuple[int, int] = [1, 1],
        num_groups: int = 8,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            if use_norm
            else nn.Identity()
        )

        # hard coded padding for now
        padding = [(k - 1) // 2 for k in kernel_size]

        self.activation = nn.SiLU()
        self.project = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)


class ResnetBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int] = [3, 3],
        stride: Tuple[int] = [1, 1],
        padding: Tuple[int] = [1, 1],
        dilation: Tuple[int] = [1, 1],
        use_norm: bool = True,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            use_norm=use_norm,
            num_groups=num_groups,
        )

        self.block2 = ConvBlock2d(
            in_channels=out_channels,
            out_channels=out_channels,
            use_norm=use_norm,
            num_groups=num_groups,
        )

        self.to_out = (
            Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.to_out(x)


class DownsampleBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: Tuple[int, int],
        num_groups: int,
        num_layers: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        super().__init__()

        if all([fac == 1 for fac in factor]):
            self.downsample = nn.Identity()
        else:
            self.downsample = Downsample2d(
                in_channels=in_channels, out_channels=out_channels, factor=factor
            )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    num_groups=num_groups,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class UpsampleBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: Tuple[int, int],
        num_groups: int,
        num_layers: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        pads: Tuple[int, int],
    ):
        super().__init__()

        self.upsample = Upsample2d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            pads=pads,
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    num_groups=num_groups,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        return x


"""Encoder"""


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        pads: Sequence[int],
        ch_matching: Sequence[int] | None = None,
        resnet_groups: int = 8,
        state: str | None = None,
    ):
        super().__init__()
        self.num_layers = len(kernel_sizes)
        # self.downsample_factor = patch_size * prod(factors)
        self.out_channels = channels * multipliers[-1]

        assert len(factors) == self.num_layers and len(num_blocks) == self.num_layers

        self.to_in = ConvBlock2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            use_norm=False,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                DownsampleBlock2d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    factor=factor,
                    num_groups=resnet_groups,
                    num_layers=blocks,
                    kernel_size=kernel,
                    stride=stride,
                )
                for i, (kernel, stride, blocks, factor) in enumerate(
                    zip(kernel_sizes[1:], strides[1:], num_blocks[1:], factors[1:])
                )
            ]
        )

        if ch_matching is not None:
            self.to_out = nn.Conv2d(ch_matching[0], ch_matching[1], 1, 1)
            # self.to_out = nn.Linear(ch_matching[0], ch_matching[1])
        else:
            self.to_out = nn.Identity()

        if state is not None:
            with open(state, "rb") as handle:
                state = pickle.load(handle)
            self.load_state_dict(state["encoder_state"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.to_in(x)
        for block in self.encoder_blocks:
            x = block(x)

        x = self.to_out(x)

        return x


"""Decoder"""


class CNNDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        pads: Sequence[int],
        resnet_groups: int = 8,
        dec_output_channels: int | None = None,
        state: str | None = None,
    ):
        super().__init__()
        self.num_layers = len(kernel_sizes)
        # self.downsample_factor = patch_size * prod(factors)
        self.out_channels = channels * multipliers[-1]

        assert len(factors) == self.num_layers and len(num_blocks) == self.num_layers

        self.decoder_blocks = nn.ModuleList(
            [
                UpsampleBlock2d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    factor=factor,
                    num_groups=resnet_groups,
                    num_layers=blocks,
                    kernel_size=kernel,
                    stride=stride,
                    pads=pad,
                )
                for i, (kernel, stride, blocks, factor, pad) in enumerate(
                    zip(
                        kernel_sizes[:-1],
                        strides[:-1],
                        num_blocks[:-1],
                        factors[:-1],
                        pads[:-1],
                    )
                )
            ]
        )

        # use a specific number of decoder output channels (different from enc)
        if dec_output_channels is not None:
            out_channels = dec_output_channels

        self.to_out = nn.Conv2d(
            in_channels=channels * multipliers[-1],
            out_channels=out_channels,
            kernel_size=kernel_sizes[-1],
            stride=strides[-1],
            padding=[(k - 1) // 2 for k in kernel_sizes[-1]],
        )

        if state is not None:
            with open(state, "rb") as handle:
                state = pickle.load(handle)
            self.load_state_dict(state["decoder_state"])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.decoder_blocks:
            x = block(x)
        # x = x.flatten(start_dim=1)
        x = self.to_out(x)
        return x
