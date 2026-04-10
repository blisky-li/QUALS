import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import prepare_4d_attention_mask, prepare_4d_causal_attention_mask, xfold
from .enc_dec_layers import SwinStageEncoder, SwinStageDecoder, SwinPatchMerging, SwinPatchExpanding
from typing import Optional, Union


class BaseModel(nn.Module):
    def __init__(self, config):
        """
        Input:
            - config: shape object; default: required.
        Method:
            - Executes the `BaseModel.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        """
        Input:
            - module: shape object; default: required.
        Method:
            - Executes the `BaseModel._init_weights` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def init_weights(self):
        """
        Input:
            - No external inputs beyond module state.
        Method:
            - Executes the `BaseModel.init_weights` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        self.apply(self._init_weights)

    def _update_full_mask(
            self,
            attention_mask: Union[torch.Tensor, None],
            inputs_embeds: torch.Tensor,
    ):
        """
        Input:
            - attention_mask (Union[torch.Tensor, None]): shape [*]; default: required.
            - inputs_embeds (torch.Tensor): shape [*]; default: required.
        Method:
            - Executes the `BaseModel._update_full_mask` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        attention_mask = prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
        # print(attention_mask.shape)
        return attention_mask

    def _update_causal_mask(
            self,
            attention_mask: Union[torch.Tensor, None],
            input_shape: torch.Size,
            inputs_embeds: torch.Tensor,
            past_key_values_length: int,
    ):
        """
        Input:
            - attention_mask (Union[torch.Tensor, None]): shape [*]; default: required.
            - input_shape (torch.Size): shape [N]; default: required.
            - inputs_embeds (torch.Tensor): shape [*]; default: required.
            - past_key_values_length (int): shape []; default: required.
        Method:
            - Executes the `BaseModel._update_causal_mask` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        attention_mask = prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        return attention_mask

    def _update_cross_attn_mask(
            self,
            encoder_hidden_states: Union[torch.Tensor, None],
            encoder_attention_mask: Union[torch.Tensor, None],
            input_shape: torch.Size,
            inputs_embeds: torch.Tensor,
    ):
        """
        Input:
            - encoder_hidden_states (Union[torch.Tensor, None]): shape [*]; default: required.
            - encoder_attention_mask (Union[torch.Tensor, None]): shape [*]; default: required.
            - input_shape (torch.Size): shape [N]; default: required.
            - inputs_embeds (torch.Tensor): shape [*]; default: required.
        Method:
            - Executes the `BaseModel._update_cross_attn_mask` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        encoder_attention_mask = prepare_4d_attention_mask(
            encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )

        return encoder_attention_mask


class SwinEncoder(nn.Module):
    def __init__(self, config):
        """
        Input:
            - config: shape object; default: required.
        Method:
            - Executes the `SwinEncoder.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        grid_size = (config.context_length, 0)
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths), device="cpu")]
        self.layers = nn.ModuleList(
            [
                SwinStageEncoder(
                    config=config,
                    dim=int(config.embed_dim),
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=SwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[tuple]:
        """
        Input:
            - hidden_states (torch.Tensor): shape [B, T, C]; default: required.
            - input_dimensions (tuple[int, int]): shape [2], usually [num_patches, 1]; default: required.
            - head_mask (torch.Tensor | None): shape [num_layers, ...] or None; default: None.
            - output_attentions (bool | None): shape []; default: None.
            - output_hidden_states (bool | None): shape []; default: None.
            - output_hidden_states_before_downsampling (bool): shape []; default: False.
            - always_partition (bool): shape []; default: False.
            - return_dict (bool): shape []; default: True.
        Method:
            - Runs the hierarchical Swin encoder stage by stage and collects token features from fine to coarse resolution.
        Output:
            - return: encoder outputs containing the final hidden states and, when requested, intermediate multiscale hidden states.
        """
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, input_dimensions, output_attentions, always_partition)

            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]
            # print(i, len(all_hidden_states),output_hidden_states_before_downsampling)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return  (hidden_states, all_hidden_states, all_self_attentions, all_reshaped_hidden_states)




class SwinDecoder(nn.Module):
    """
    Decoder mirror of SwinEncoder.
    Total stages = num_layers + 1  (for symmetric embedding-level reconstruction)
    """

    def __init__(self, config):
        """
        Input:
            - config: shape object; default: required.
        Method:
            - Executes the `SwinDecoder.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()

        self.config = config
        self.num_encoder_layers = len(config.depths)
        self.num_layers = self.num_encoder_layers + 1  # 例如 4 + 1 = 5

        # 计算 encoder 的 total depth（用于 drop_path 划分）
        encoder_total_depth = sum(config.depths)

        # decoder 的 total depth = encoder_total_depth + 1（额外一层）
        decoder_total_depth = encoder_total_depth + 1

        # drop_path reversed
        dpr = [x.item() for x in torch.linspace(
            0, config.drop_path_rate, decoder_total_depth, device="cpu"
        )]

        # ↓ 与 encoder 严格一致的层注册方式
        self.layers = nn.ModuleList()

        dp_index = 0
        grid_size = (config.context_length, 0)

        # --------------------------
        # 0~num_encoder_layers-1 层
        # 与 encoder 对称（倒序）
        # --------------------------
        

        for i_layer in range(self.num_encoder_layers):

            # encoder: (grid // 2^i)
            # decoder: reverse resolution
            mirror_idx = self.num_encoder_layers - 1 - i_layer

            input_resolution = (
                grid_size[0] // (2 ** mirror_idx),
                0
            )

            depth = config.depths[mirror_idx]
            dp_slice = dpr[dp_index: dp_index + depth]
            dp_index += depth

            num_heads = config.num_heads[mirror_idx]
            dim = int(config.embed_dim)

            # except the LAST layer, all decoder layers should upsample
            upsample = SwinPatchExpanding if (i_layer < self.num_encoder_layers - 1) else None

            stage = SwinStageDecoder(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                depth=depth,
                num_heads=num_heads,
                drop_path=dp_slice,
                upsample=upsample,
            )

            self.layers.append(stage)

        # --------------------------
        # 最后一层（embedding-level）
        # input_resolution 与上一层一致
        # depth = 1
        # heads = config.num_heads[0]
        # upsample = None
        # --------------------------
        final_input_resolution = (
            grid_size[0] // (2 ** 0),
            0
        )

        final_depth = 1
        final_dp_slice = dpr[dp_index: dp_index + final_depth]

        final_stage = SwinStageDecoder(
            config=config,
            dim=int(config.embed_dim),
            input_resolution=final_input_resolution,
            depth=final_depth,
            num_heads=config.num_heads[0],
            drop_path=final_dp_slice,
            upsample=None,
        )

        self.layers.append(final_stage)
        self.outlayer = nn.Linear(int(config.embed_dim), int(config.patch_size))

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,           # start from VQ coarse latent
        input_dimensions,        # (H,W)
        vq_latents,              # len = num_decoder_layers
        output_attentions=False,
        output_hidden_states=False,
        always_partition: Optional[bool] = False,
        return_dict=True,
    ):
        """
        Input:
            - hidden_states (torch.Tensor): shape [B, T_coarse, C]; default: required.
            - input_dimensions (tuple[int, int]): shape [2], usually [T_coarse, 1]; default: required.
            - head_mask (torch.Tensor | None): shape [num_layers, ...] or None; default: None.
            - output_attentions (bool | None): shape []; default: None.
            - output_hidden_states (bool | None): shape []; default: None.
            - return_dict (bool): shape []; default: True.
            - vq_latents (list[torch.Tensor] | None): shapes like [[B, T_i, C], ...]; default: None.
        Method:
            - Starts from the coarsest quantized tokens and progressively upsamples while injecting multilevel VQ latents into each decoder stage.
        Output:
            - return: decoder outputs whose first element is the reconstructed patch token sequence of shape [B, T_patch, patch_size].
        """
        all_hidden_states = () if output_hidden_states else None
        all_attn = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):

            kv_states = vq_latents[i]  # latent for this stage

            if hidden_states.shape[1] > kv_states.shape[1]:
                
                L_kv = kv_states.shape[1]
                hidden_states = hidden_states[:, :L_kv, :]
                input_dimensions = (hidden_states.shape[1], input_dimensions[1])

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                kv_states=kv_states,
                input_dimensions=input_dimensions,
                output_attentions=output_attentions,
                always_partition =  always_partition
            )

            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if output_attentions:
                all_attn += (layer_outputs[2:],)

        hidden_states = self.outlayer(hidden_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attn] if v is not None)

        return hidden_states, all_hidden_states, all_attn


def xfoldold(x_unfolded, patch, step):
    """
    Input:
        - x_unfolded: shape [*]; default: required.
        - patch: shape []; default: required.
        - step: shape []; default: required.
    Method:
        - Executes the `xfoldold` logic using the provided inputs and current module state.
    Output:
        - return: value(s) produced by this helper; exact shape follows the implementation below.
    """
    B, C, K, N = x_unfolded.shape
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

def xfold(x_unfolded, patch, step, eps=1e-6):
    """
    Input:
        - x_unfolded (torch.Tensor): shape [B, C, patch, num_patches]; default: required.
        - patch (int): shape []; default: required.
        - step (int): shape []; default: required.
        - eps (float): shape []; default: 1e-6.
    Method:
        - Overlap-adds patch-wise decoder outputs back to the original temporal axis and normalizes overlapped positions.
    Output:
        - x (torch.Tensor): shape [B, C, L], where `L = step * (num_patches - 1) + patch`.
    """
    B, C, K, N = x_unfolded.shape
    L = (N - 1) * step + patch

    x_unfolded_reshape = x_unfolded.reshape(B, C*patch, -1)
    fold = torch.nn.Fold(output_size=(1, L), kernel_size=(1, patch), stride=(1, step))
    x_fold = fold(x_unfolded_reshape)

    ones = torch.ones((B, C, L), device=x_unfolded.device, dtype=x_unfolded.dtype)
    ones_unfolded = ones.unfold(dimension=-1, size=patch, step=step).transpose(-1, -2).reshape(B, C*patch, -1)
    overlap = fold(ones_unfolded)

    # 防止除 0
    overlap = torch.clamp(overlap, min=eps)

    x_reconstructed = x_fold / overlap
    x_reconstructed = x_reconstructed.squeeze(-2)
    return x_reconstructed