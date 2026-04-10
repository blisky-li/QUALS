import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_layer import SwinAttention, window_partition1D, window_reverse1D, SwinDropPath
from transformers.activations import ACT2FN
from transformers.utils import logging
from typing import Optional, Union
from functools import partial

def torch_int(x):
    """
    Input:
        - x: shape [*]; default: required.
    Method:
        - Executes the `torch_int` logic using the provided inputs and current module state.
    Output:
        - return: value(s) produced by this helper; exact shape follows the implementation below.
    """

    return x.to(torch.int64) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)


class SwinIntermediate(nn.Module):
    def __init__(self, config, dim):
        """
        Input:
            - config: shape object; default: required.
            - dim: shape []; default: required.
        Method:
            - Executes the `SwinIntermediate.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Input:
            - hidden_states (torch.Tensor): shape [*]; default: required.
        Method:
            - Executes the `SwinIntermediate.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SwinOutput(nn.Module):
    def __init__(self, config, dim):
        """
        Input:
            - config: shape object; default: required.
            - dim: shape []; default: required.
        Method:
            - Executes the `SwinOutput.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Input:
            - hidden_states (torch.Tensor): shape [*]; default: required.
        Method:
            - Executes the `SwinOutput.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class SwinLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, drop_path_rate=0.0, shift_size=0, mode=None):
        """
        Input:
            - config: shape object; default: required.
            - dim: shape []; default: required.
            - input_resolution: shape [N]; default: required.
            - num_heads: shape []; default: required.
            - drop_path_rate: shape [*]; default: 0.0.
            - shift_size: shape [*]; default: 0.
            - mode: shape [*]; default: None.
        Method:
            - Executes the `SwinLayer.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mode = mode
        
        self.attention = SwinAttention(config, dim, num_heads, window_size=self.window_size)
        

        self.drop_path = SwinDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.intermediate = SwinIntermediate(config, dim)
        self.output = SwinOutput(config, dim)

    def set_shift_and_window_size(self, input_resolution):
        """
        Input:
            - input_resolution: shape [N]; default: required.
        Method:
            - Executes the `SwinLayer.set_shift_and_window_size` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        if min(input_resolution) <= self.window_size:
            
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = torch_int(0)
            #self.window_size = (torch.min(torch.tensor(input_resolution)) if torch.jit.is_tracing() else min(input_resolution))

    def get_attn_mask(self, height, width, dtype, device):
        """
        Input:
            - height: shape []; default: required.
            - width: shape []; default: required.
            - dtype: shape [*]; default: required.
            - device: shape [*]; default: required.
        Method:
            - Executes the `SwinLayer.get_attn_mask` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype, device=device)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition1D(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        # 只考虑 height（序列长度）方向的 padding
        """
        Input:
            - hidden_states: shape [*]; default: required.
            - height: shape []; default: required.
            - width: shape []; default: required.
        Method:
            - Executes the `SwinLayer.maybe_pad` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size

        # 不对 width 做 pad：pad_w_left = pad_w_right = 0
        pad_values = (0, 0, 0, 0, 0, pad_bottom)
        if pad_bottom > 0:
            hidden_states = nn.functional.pad(hidden_states, pad_values)

        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        kv_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            - hidden_states (torch.Tensor): shape [B, T, C]; default: required.
            - input_dimensions (tuple[int, int]): shape [2]; default: required.
            - head_mask (torch.FloatTensor | None): shape [num_heads] or None; default: None.
            - output_attentions (bool): shape []; default: False.
            - always_partition (bool): shape []; default: False.
        Method:
            - Applies shifted-window attention, residual connections, and the feed-forward block for one Swin layer.
        Output:
            - return: tuple whose first element is the updated hidden state tensor of shape [B, T, C].
        """
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass

        
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        # print('encoder.shape: ',hidden_states.shape, input_dimensions)

        hidden_states = self.layernorm_before(hidden_states)

        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # print('shifted_hidden_states.shape: ', shifted_hidden_states.shape, self.shift_size)

        # partition windows
        hidden_states_windows = window_partition1D(shifted_hidden_states, self.window_size)
        # print('hidden windows ori.shape: ', hidden_states_windows.shape)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * 1, channels)
        attn_mask = self.get_attn_mask(
            height_pad, width_pad, dtype=hidden_states.dtype, device=hidden_states_windows.device
        )
        # print('hidden windows.shape: ', hidden_states_windows.shape)
        if self.mode == "encoder":
            attention_outputs = self.attention(hidden_states = hidden_states_windows,
                                            kv_states = None,
                                            attention_mask = attn_mask,
                                            output_attentions = output_attentions)
        else:
            # 需要将 kv_states 也切成 window（你已实现 window_partition1D）
            kv = kv_states.view(batch_size, height, width, channels)
            kv, _ = self.maybe_pad(kv, height, width)
            kv = window_partition1D(kv, self.window_size).view(-1, self.window_size * 1, channels)

            attention_outputs = self.attention(hidden_states = hidden_states_windows,
                                            kv_states = kv,
                                            attention_mask = attn_mask,
                                            output_attentions = output_attentions)

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(-1, self.window_size, 1, channels)
        shifted_windows = window_reverse1D(attention_windows, self.window_size, height_pad)

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        hidden_states = shortcut + self.drop_path(attention_windows)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs


class SwinPatchMerging2D(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        """
        Input:
            - input_resolution (tuple[int]): shape []; default: required.
            - dim (int): shape []; default: required.
            - norm_layer (nn.Module): shape [*]; default: nn.LayerNorm.
        Method:
            - Executes the `SwinPatchMerging2D.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def maybe_pad(self, input_feature, height, width):
        """
        Input:
            - input_feature: shape [*]; default: required.
            - height: shape []; default: required.
            - width: shape []; default: required.
        Method:
            - Executes the `SwinPatchMerging2D.maybe_pad` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: tuple[int, int]) -> torch.Tensor:
        """
        Input:
            - input_feature (torch.Tensor): shape [*]; default: required.
            - input_dimensions (tuple[int, int]): shape []; default: required.
        Method:
            - Executes the `SwinPatchMerging2D.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape
        
        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be divisible by width and height, if needed
        # print('Merge shape:', input_feature.shape)
        input_feature = self.maybe_pad(input_feature, height, width)
        # print('Merge pad shape:', input_feature.shape)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # batch_size height/2 width/2 4*num_channels
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature

class SwinPatchMerging(nn.Module):
    """
    1D Patch Merging for time series.
    Downsamples only the H dimension (sequence length).
    W dimension is fixed to 1.
    """

    def __init__(self, input_length: int, dim: int, norm_layer=nn.LayerNorm):
        """
        Input:
            - input_length (int): shape []; default: required.
            - dim (int): shape []; default: required.
            - norm_layer: shape [*]; default: nn.LayerNorm.
        Method:
            - Executes the `SwinPatchMerging.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.input_length = input_length  # H
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def maybe_pad(self, x, H):
        """
        Input:
            - x: shape [*]; default: required.
            - H: shape []; default: required.
        Method:
            - Executes the `SwinPatchMerging.maybe_pad` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        if H % 2 == 1:
            # pad one element on bottom only
            pad = (0, 0, 0, 0, 0, 1)  # pad H dimension (last pair)
            x = F.pad(x, pad)
        return x

    def forward(self, x, input_dimensions):
        """
        Input:
            - x: shape [*]; default: required.
            - input_dimensions: shape [N]; default: required.
        Method:
            - Executes the `SwinPatchMerging.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """
        H, W = input_dimensions
        B, HW, C = x.shape
        assert W == 1

        # reshape to [B, H, 1, C]
        x = x.view(B, H, 1, C)

        # pad if H not divisible by 2
        x = self.maybe_pad(x, H)
        H2 = x.shape[1] // 2   # H2 = H/2

        # split even/odd positions
        x_even = x[:, 0::2, :, :]   # [B, H2, 1, C]
        x_odd  = x[:, 1::2, :, :]   # [B, H2, 1, C]

        # concat along channel
        x = torch.cat([x_even, x_odd], dim=-1)  # [B, H2, 1, 2C]

        x = x.view(B, H2, 2 * C)

        x = self.norm(x)
        x = self.reduction(x)  # [B, H2, C]

        return x


class SwinPatchExpanding(nn.Module):
    """
    1D version: only height (sequence L) changes, width = 1 fixed.
    Decoder counterpart of 1D SwinPatchMerging.
    """

    def __init__(self, input_length, dim, norm_layer=nn.LayerNorm):
        """
        Input:
            - input_length: shape [*]; default: required.
            - dim: shape []; default: required.
            - norm_layer: shape [*]; default: nn.LayerNorm.
        Method:
            - Executes the `SwinPatchExpanding.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.input_length = input_length   # L/2
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)  # 1D: split into 2 patches
        self.norm = norm_layer(dim)

    def forward(self, x, input_dimensions):
        """
        Input:
            - x: shape [*]; default: required.
            - input_dimensions: shape [N]; default: required.
        Method:
            - Executes the `SwinPatchExpanding.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """

        H, W = input_dimensions
        B, L2, C = x.shape
        # assert L2 == self.input_length

        x = self.norm(x)
        x = self.expand(x)    # [B, L/2, 2C]

        # reshape: split into 2 tokens for each position
        x = x.view(B, L2, 2, C)

        # reorder to length dimension
        x = x.view(B, L2 * 2, C)

        return x



logger = logging.get_logger(__name__)

class GradientCheckpointingLayer(nn.Module):
    """Base class for layers with gradient checkpointing.

    This class enables gradient checkpointing functionality for a layer. By default, gradient checkpointing is disabled
    (`gradient_checkpointing = False`). When `model.set_gradient_checkpointing()` is called, gradient checkpointing is
    enabled by setting `gradient_checkpointing = True` and assigning a checkpointing function to `_gradient_checkpointing_func`.

    Important:

        When using gradient checkpointing with `use_reentrant=True`, inputs that require gradients (e.g. hidden states)
        must be passed as positional arguments (`*args`) rather than keyword arguments to properly propagate gradients.

        Example:

            ```python
            >>> # Correct - hidden_states passed as positional arg
            >>> out = self.layer(hidden_states, attention_mask=attention_mask)

            >>> # Incorrect - hidden_states passed as keyword arg
            >>> out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask)
            ```
    """

    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        """
        Input:
            - *args: shape [*]; default: variadic positional input.
            - **kwargs: shape [*]; default: variadic keyword input.
        Method:
            - Executes the `GradientCheckpointingLayer.__call__` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """
        if self.gradient_checkpointing and self.training:
            do_warn = False
            layer_name = self.__class__.__name__
            message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

            if "use_cache" in kwargs and kwargs["use_cache"]:
                kwargs["use_cache"] = False
                message += " `use_cache=False`,"
                do_warn = True

            # different names for the same thing in different layers
            # TODO cyril: this one without `S` can be removed after deprection cycle
            if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                kwargs["past_key_value"] = None
                message += " `past_key_value=None`,"
                do_warn = True

            if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                kwargs["past_key_values"] = None
                message += " `past_key_values=None`,"
                do_warn = True

            if "layer_past" in kwargs and kwargs["layer_past"] is not None:
                kwargs["layer_past"] = None
                message += " `layer_past=None`,"
                do_warn = True

            # warn if anything was changed
            if do_warn:
                message = message.rstrip(",") + "."
                logger.warning_once(message)

            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)

class SwinStageEncoder(GradientCheckpointingLayer):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        """
        Input:
            - config: shape object; default: required.
            - dim: shape []; default: required.
            - input_resolution: shape [N]; default: required.
            - depth: shape [*]; default: required.
            - num_heads: shape []; default: required.
            - drop_path: shape [*]; default: required.
            - downsample: shape [*]; default: required.
        Method:
            - Executes the `SwinStageEncoder.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.config = config
        self.dim = dim
        
        self.blocks = nn.ModuleList(
            [
                SwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    drop_path_rate=drop_path[i],
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                    mode = 'encoder'
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> tuple[torch.Tensor]:
        """
        Input:
            - hidden_states (torch.Tensor): shape [*]; default: required.
            - input_dimensions (tuple[int, int]): shape []; default: required.
            - output_attentions (Optional[bool]): shape []; default: False.
            - always_partition (Optional[bool]): shape []; default: False.
        Method:
            - Executes the `SwinStageEncoder.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """
        height, width = input_dimensions

        # print('encoder layer: ', hidden_states.shape)


        for i, layer_module in enumerate(self.blocks):
            layer_outputs = layer_module(hidden_states, input_dimensions, output_attentions, always_partition)

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class SwinStageDecoder(GradientCheckpointingLayer):
    """
    Mirror of encoder SwinStage.
    """

    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, upsample):
        """
        Input:
            - config: shape object; default: required.
            - dim: shape []; default: required.
            - input_resolution: shape [N]; default: required.
            - depth: shape [*]; default: required.
            - num_heads: shape []; default: required.
            - drop_path: shape [*]; default: required.
            - upsample: shape [*]; default: required.
        Method:
            - Executes the `SwinStageDecoder.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.config = config
        self.dim = dim

        # decoder blocks
        self.blocks = nn.ModuleList(
            [
                SwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    drop_path_rate=drop_path[i],
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                    mode='decoder'
                )
                for i in range(depth)
            ]
        )

        # upsample layer (symmetric to SwinPatchMerging)
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        output_attentions: bool = False,
        always_partition: bool = False,
    ):

        # print('Decoder Stage Shape: ', hidden_states.shape, kv_states.shape, input_dimensions)


        """
        Input:
            - hidden_states (torch.Tensor): shape [*]; default: required.
            - kv_states (torch.Tensor): shape [*]; default: required.
            - input_dimensions (tuple[int, int]): shape []; default: required.
            - output_attentions (bool): shape []; default: False.
            - always_partition (bool): shape []; default: False.
        Method:
            - Executes the `SwinStageDecoder.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """
        height, width = input_dimensions

        for block in self.blocks:
            out = block(
                hidden_states=hidden_states,
                kv_states=kv_states,
                input_dimensions=input_dimensions,
                output_attentions=output_attentions,
                always_partition=always_partition,
            )
            hidden_states = out[0]

        hidden_states_before_upsample = hidden_states

        if self.upsample is not None:
            # resolution doubles (H*2)
            height_new = height * 2
            width_new = width
            output_dimensions = (height, width, height_new, width_new)
            hidden_states = self.upsample(hidden_states_before_upsample, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, hidden_states_before_upsample, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


