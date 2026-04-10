import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union, List
import math
import collections.abc


def meshgrid(*tensors: Union[torch.Tensor, List[torch.Tensor]], indexing: str = "ij"):
    """
    Input:
        - *tensors: shape [D1, ..., Dn] for each input tensor; default: required.
        - indexing (str): shape []; default: "ij".
    Method:
        - Wraps `torch.meshgrid` with an explicit `indexing` argument so 1D Swin window coordinates are generated consistently.
    Output:
        - return: tuple of coordinate tensors with the same broadcastable spatial shape as the input tensors.
    """
    return torch.meshgrid(*tensors, indexing=indexing)

def window_partition1D(input_feature: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Input:
        - input_feature (torch.Tensor): shape [B, H, C]; default: required.
        - window_size (int): shape []; default: required.
    Method:
        - Splits the 1D token axis `H` into non-overlapping windows of length `window_size` for local self-attention.
    Output:
        - windows (torch.Tensor): shape [B * num_windows, window_size, C].
    """
    B, H, W, C = input_feature.shape
    # print(B,H,W,C,window_size)
    assert W == 1, "This 1D version only supports W=1 inputs."

    assert H % window_size == 0, f"H={H} must be divisible by window_size={window_size}."

    num_windows = H // window_size
    # reshape to [B, num_windows, window_size, 1, C]
    windows = input_feature.view(B, num_windows, window_size, 1, C)

    # merge batch and window index -> [B*num_windows, window_size, 1, C]
    windows = windows.reshape(B * num_windows, window_size, 1, C)

    return windows

def window_reverse1D(windows, window_size, H):
    """
    Input:
        - windows (torch.Tensor): shape [B * num_windows, window_size, C]; default: required.
        - window_size (int): shape []; default: required.
        - H (int): shape []; default: required.
    Method:
        - Reassembles partitioned local windows back into the original 1D token order.
    Output:
        - input_feature (torch.Tensor): shape [B, H, C].
    """
    B_times_win, window_h, _, C = windows.shape
    assert window_h == window_size, "window_size mismatch."

    num_windows = H // window_size
    B = B_times_win // num_windows

    # reshape back: [B, num_windows, window_size, 1, C]
    windows = windows.view(B, num_windows, window_size, 1, C)

    # merge windows back: [B, H, 1, C]
    input_feature = windows.reshape(B, H, 1, C)

    return input_feature


class SwinAttentionBlock(nn.Module):
    """
    Unified Swin attention block.
    If kv_states is None → self-attention
    If kv_states is not None → cross-attention
    """
    def __init__(self, config, dim, num_heads, window_size):
        """
        Input:
            - config: shape object; default: required.
            - dim: shape []; default: required.
            - num_heads: shape []; default: required.
            - window_size: shape []; default: required.
        Method:
            - Executes the `SwinAttentionBlock.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.all_head_dim = dim

        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable)
            else (window_size, 1)
        )

        # --- Relative Position Bias ---
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2*self.window_size[0]-1) * (2*self.window_size[1]-1),
                num_heads
            )
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0] += self.window_size[0]-1
        relative_coords[:,:,1] += self.window_size[1]-1
        relative_coords[:,:,0] *= (2*self.window_size[1]-1)
        self.register_buffer(
            "relative_position_index",
            relative_coords.sum(-1)
        )

        # --- Q/K/V projections (shared for self and cross) ---
        self.query = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.key   = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.value = nn.Linear(dim, dim, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states,          # Q
        kv_states=None,         # KV (None → self-attn)
        attention_mask=None,
        output_attentions=False
    ):
        """
        Input:
            - hidden_states: shape [*]; default: required.
            - kv_states: shape [*]; default: None.
            - attention_mask: shape [*]; default: None.
            - output_attentions: shape [*]; default: False.
        Method:
            - Executes the `SwinAttentionBlock.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """

        # ----------------------
        # 1. Q projection
        # ----------------------
        B, Lq, C = hidden_states.shape
        q = self.query(hidden_states)
        q = q.view(B, Lq, self.num_heads, self.head_dim).permute(0,2,1,3)

        # ----------------------
        # 2. K/V projection
        # ----------------------


        if kv_states is None:
            kv_states = hidden_states   # self-attn
        '''else:
            print('Cross Attention ! ')'''
        
        B, Lkv, _ = kv_states.shape
        k = self.key(kv_states).view(B, Lkv, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.value(kv_states).view(B, Lkv, self.num_heads, self.head_dim).permute(0,2,1,3)

        # ----------------------
        # 3. Attention scores
        # ----------------------
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # --- relative position bias ---
        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(
            self.window_size[0]*self.window_size[1],
            self.window_size[0]*self.window_size[1],
            -1
        )
        rel_bias = rel_bias.permute(2,0,1).unsqueeze(0)
        attn_scores = attn_scores + rel_bias

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # ----------------------
        # 4. Softmax
        # ----------------------
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # ----------------------
        # 5. Attention output
        # ----------------------
        context = torch.matmul(attn_probs, v)
        context = context.permute(0,2,1,3).contiguous().view(B, Lq, C)

        return (context, attn_probs) if output_attentions else (context,)
    
    
class SwinAttentionOutput(nn.Module):
    def __init__(self, config, dim):
        """
        Input:
            - config: shape object; default: required.
            - dim: shape []; default: required.
        Method:
            - Executes the `SwinAttentionOutput.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input:
            - hidden_states (torch.Tensor): shape [*]; default: required.
            - input_tensor (torch.Tensor): shape [*]; default: required.
        Method:
            - Executes the `SwinAttentionOutput.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class SwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        """
        Input:
            - config: shape object; default: required.
            - dim: shape []; default: required.
            - num_heads: shape []; default: required.
            - window_size: shape []; default: required.
        Method:
            - Executes the `SwinAttention.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.block = SwinAttentionBlock(config, dim, num_heads, window_size)
        self.output = SwinAttentionOutput(config, dim)

    def forward(
        self,
        hidden_states,       # Q
        kv_states=None,      # None = self-attention
        attention_mask=None,
        output_attentions=False
    ):
        """
        Input:
            - hidden_states: shape [*]; default: required.
            - kv_states: shape [*]; default: None.
            - attention_mask: shape [*]; default: None.
            - output_attentions: shape [*]; default: False.
        Method:
            - Executes the `SwinAttention.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """
        attn_outputs = self.block(
            hidden_states,
            kv_states=kv_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        context = attn_outputs[0]

        # 这里保持你原样：output(context, hidden_states)
        projected = self.output(context, hidden_states)

        return (projected,) + attn_outputs[1:]



def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Input:
        - input (torch.Tensor): shape [*]; default: required.
        - drop_prob (float): shape []; default: 0.0.
        - training (bool): shape []; default: False.
    Method:
        - Executes the `drop_path` logic using the provided inputs and current module state.
    Output:
        - return: value(s) produced by this helper; exact shape follows the implementation below.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Swin
class SwinDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        """
        Input:
            - drop_prob (Optional[float]): shape []; default: None.
        Method:
            - Executes the `SwinDropPath.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Input:
            - hidden_states (torch.Tensor): shape [*]; default: required.
        Method:
            - Executes the `SwinDropPath.forward` logic using the provided inputs and current module state.
        Output:
            - return: tensor(s) or structured outputs produced by this module; exact layout follows the implementation below.
        """
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """
        Input:
            - No external inputs beyond module state.
        Method:
            - Executes the `SwinDropPath.extra_repr` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        return f"p={self.drop_prob}"