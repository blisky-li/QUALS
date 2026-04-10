import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Examples:

    ```python
    >>> import torch
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(torch.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=torch.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool = False, sliding_window: Optional[int] = None):
        """
        Input:
            - is_causal (bool): shape []; default: False.
            - sliding_window (Optional[int]): shape []; default: None.
        Method:
            - Executes the `AttentionMaskConverter.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Input:
            - attention_mask_2d (torch.Tensor): shape [*]; default: required.
            - query_length (int): shape []; default: required.
            - dtype (torch.dtype): shape [*]; default: required.
            - key_value_length (Optional[int]): shape []; default: None.
        Method:
            - Executes the `AttentionMaskConverter.to_4d` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )

        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask
    
    def to_causal_4d(
        self,
        batch_size: int,
    query_length: int,
    key_value_length: int,
    dtype: torch.dtype,
    device: Union[torch.device, "str"] = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        Input:
            - batch_size (int): shape []; default: required.
            - query_length (int): shape []; default: required.
            - key_value_length (int): shape []; default: required.
            - dtype (torch.dtype): shape [*]; default: required.
            - device (Union[torch.device, "str"]): shape []; default: "cpu".
        Method:
            - Executes the `AttentionMaskConverter.to_causal_4d` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # If shape is not cached, create a new causal mask and cache it
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Input:
            - mask (torch.Tensor): shape [*]; default: required.
            - dtype (torch.dtype): shape [*]; default: required.
            - tgt_len (Optional[int]): shape []; default: None.
        Method:
            - Executes the `AttentionMaskConverter._expand_mask` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _make_causal_mask(
            input_ids_shape: torch.Size,
            dtype: torch.dtype,
            device: torch.device,
            past_key_values_length: int = 0,
            sliding_window: Optional[int] = None,
    ):
        """
        Input:
            - input_ids_shape (torch.Size): shape [N]; default: required.
            - dtype (torch.dtype): shape [*]; default: required.
            - device (torch.device): shape [*]; default: required.
            - past_key_values_length (int): shape []; default: 0.
            - sliding_window (Optional[int]): shape []; default: None.
        Method:
            - Executes the `AttentionMaskConverter._make_causal_mask` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1

            context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
            # Recent changes in PyTorch prevent mutations on tensors converted with aten::_to_copy
            # See https://github.com/pytorch/pytorch/issues/127571
            if is_torchdynamo_compiling():
                mask = mask.clone()
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Input:
        - mask (torch.Tensor): shape [*]; default: required.
        - dtype (torch.dtype): shape [*]; default: required.
        - tgt_len (Optional[int]): shape []; default: None.
    Method:
        - Executes the `prepare_4d_attention_mask` logic using the provided inputs and current module state.
    Output:
        - return: value(s) produced by this helper; exact shape follows the implementation below.
    """
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, tuple, list],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Input:
        - attention_mask (Optional[torch.Tensor]): shape [*]; default: required.
        - input_shape (Union[torch.Size, tuple, list]): shape [N]; default: required.
        - inputs_embeds (torch.Tensor): shape [*]; default: required.
        - past_key_values_length (int): shape []; default: required.
        - sliding_window (Optional[int]): shape []; default: None.
    Method:
        - Executes the `prepare_4d_causal_attention_mask` logic using the provided inputs and current module state.
    Output:
        - return: value(s) produced by this helper; exact shape follows the implementation below.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


def xfold(x_unfolded, patch, step):
    """
    Input:
        - x_unfolded (torch.Tensor): shape [B, C, patch, num_patches]; default: required.
        - patch (int): shape []; default: required.
        - step (int): shape []; default: required.
    Method:
        - Reconstructs a long sequence by overlap-adding 1D patches along the temporal axis.
    Output:
        - x (torch.Tensor): shape [B, C, L].
    """
    B, C, N, K = x_unfolded.shape
    # print(x_unfolded.shape)
    L = (N - 1) * step + patch
    # ============= 逆操作 =============
    # 转换成 Fold 需要的输入: (B, C*n, num_patch)
    x_unfolded_reshape = x_unfolded.reshape(B, C*patch, -1)

    # 定义Fold (把patch拼回去)
    fold = torch.nn.Fold(output_size=(1, L), kernel_size=(1, patch), stride=(1, step))

    # 输入给Fold (视为1D -> 2D)
    x_fold = fold(x_unfolded_reshape)

    # 补偿重叠部分 (用全1卷积计算覆盖次数)
    ones = torch.ones((B, C, L), device=x_unfolded.device, dtype=x_unfolded.dtype)
    ones_unfolded = ones.unfold(dimension=-1, size=patch, step=step).transpose(-1, -2).reshape(B, C*patch, -1)
    overlap = fold(ones_unfolded)

    # 最终恢复结果
    x_reconstructed = x_fold / overlap
    x_reconstructed = x_reconstructed.squeeze(-2)  # (B, C, L)
    return  x_reconstructed