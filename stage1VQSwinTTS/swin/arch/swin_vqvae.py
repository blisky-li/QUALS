from .enc_dec import BaseModel, SwinEncoder, SwinDecoder, xfold
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicts.metrics import masked_mae
from .vector_quantizer import VectorQuantizer, quantize
from .embed import SwinEmbeddings
import os
import matplotlib.pyplot as plt
from typing import Optional, Union
import numpy as np


class Swin(BaseModel):

    def __init__(self, config):
        """
        Input:
            - config: shape object; default: required.
        Method:
            - Executes the `Swin.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__(config)

        hid_dim = config['hid_dim']

        self.embeddings = SwinEmbeddings(config, use_mask_token=False)
        self.encoder = SwinEncoder(config)
        self.vqmodel = VectorQuantizer(
            dim=hid_dim,
            codebook_dim=config['vq_vae']['codebook_dim'],
            rotation_trick = config['vq_vae']['rotation_trick'],
            use_cosine_sim = config['vq_vae']['use_cosine_sim'],
            decay = config['vq_vae']['decay'],
            kmeans_init = config['vq_vae']["kmeans_init"],
            sync_codebook = config['vq_vae']["sync_codebook"],
            commitment_weight = config['vq_vae']["commitment_weight"],
            codebook_size = config['vq_vae']["codebook_size"],
        )

        self.decoder = SwinDecoder(config)

        self.head_mask = None
        self.output_attentions = None
        self.output_hidden_states = True
        self.patch_size = config.patch_size
        self.stride = config.stride
        # Initialize weights and apply final processing
        self.config = config
        self.init_weights()

    def loss(self, ground_truth, reconstruction):
        """
        Input:
            - ground_truth: shape [*]; default: required.
            - reconstruction: shape [*]; default: required.
        Method:
            - Executes the `Swin.loss` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        loss = masked_mae(prediction=reconstruction, target=ground_truth)
        return loss

    def pre_process(self, time_series):
        # inputs:
        #   - time_series: batch_size, length
        # outputs:
        #   - time_series: batch_size, length
        #   - nan_mask: batch_size, length

        # TODO: padding
        """
        Input:
            - time_series (torch.Tensor): shape [B, C, L] or [B, L]; default: required.
        Method:
            - Builds a valid-value mask from NaNs and replaces NaNs with zeros so the encoder can consume dense tensors.
        Output:
            - time_series (torch.Tensor): same shape as input, with NaNs replaced by 0.
            - nan_mask (torch.BoolTensor): same shape as `time_series`, where True marks valid values.
        """
        nan_mask = ~torch.isnan(time_series)  # NaN: 0, non-NaN: 1
        time_series = torch.nan_to_num(time_series, nan=0.0)
        return time_series, nan_mask

    def post_process(self, outputs, time_series, nan_mask):
        # outputs: [batch_size, (num_patches)*patch_size]
        # time_series: [batch_size, (num_patches)*patch_size]
        # nan_mask: [batch_size, (num_patches)*patch_size], 0 is NaN, 1 is non-NaN
        # mae_mask: [batch_size, (num_registers+num_patches)*patch_size], 0 is keep, 1 is remove
        """
        Input:
            - outputs (torch.Tensor): shape [B, C, L]; default: required.
            - time_series (torch.Tensor): shape [B, C, L]; default: required.
            - nan_mask (torch.BoolTensor): shape [B, C, L]; default: required.
        Method:
            - Restores invalid positions to NaN in both the ground truth and reconstructed tensors so metrics ignore masked locations.
        Output:
            - time_series (torch.Tensor): shape [B, C, L], with invalid positions reset to NaN.
            - outputs (torch.Tensor): shape [B, C, L], with invalid positions reset to NaN.
        """
        mask = (nan_mask).bool()  # values that are not NaN and are masked are used to calculate loss
        time_series[~mask] = torch.nan
        outputs[~mask] = torch.nan
        return time_series, outputs

    def forward(self, time_series, return_reconstruction=False, iter_num=None):
        
        # torch.autograd.set_detect_anomaly(True)
        # print(time_series.device)
        """
        Input:
            - time_series (torch.Tensor): shape [B, L] or [B, C, L], with C defaulting to 1 when omitted; default: required.
            - return_reconstruction (bool): shape []; default: False.
            - iter_num (int | None): shape []; default: None.
        Method:
            - Normalizes NaNs, embeds the sequence into Swin patch tokens, encodes multiscale latents, vector-quantizes them, and decodes them back to the reconstructed time series.
        Output:
            - when `return_reconstruction=False`: `(losses, perplexity)`.
            - when `return_reconstruction=True`: `(losses, perplexity, ground_truth, reconstruction)`, where both tensors have shape [B, C, L].
        """
        if len(time_series.shape) == 2:
            time_series = time_series.unsqueeze(1)
        time_series, nan_mask = self.pre_process(time_series)
        
        # print(self.config['vq_vae'])
        # print(time_series.shape, nan_mask.shape, torch.isnan(time_series).any(),'Transformer Input Shape')

        input_time_series = time_series.permute(0, 2, 1).unsqueeze(1)
        attention_mask = nan_mask.transpose(-1,-2).squeeze(-1)
    
        B, C, L, W= input_time_series.shape
        
        # attention_mask = torch.ones(B, C).to(input_time_series.device).bool()
        # print('input_time_series',input_time_series.shape, attention_mask.shape)

        input_time_series, output_dimensions , mask = self.embeddings(input_time_series, bool_masked_pos=None, interpolate_pos_encoding=False)
        # print(input_time_series)
        # print('embedding check: ', torch.isnan(input_time_series).any(), input_time_series.requires_grad)
        # input_time_series = input_time_series.transpose(-1, -2)

        encoder_outputs =  self.encoder(input_time_series,
            output_dimensions,
            output_attentions = self.output_attentions,
            output_hidden_states = self.output_hidden_states,
            output_hidden_states_before_downsampling = True,
            return_dict=True)

        hidden_states = encoder_outputs[1]   # list fine→coarse

        
        hidden_states = list(hidden_states)[::-1]  # 反转成 coarse→fine
        token_lengths = [h.shape[1] for h in hidden_states]
        vq_input = torch.cat(hidden_states, dim=1)
        # print('encoder output check: ', torch.isnan(vq_input).any(), vq_input.requires_grad)
        # hidden_states: layer = 4
        # B, 32, hidden: layer 4th
        # B, 64, hidden: layer 3th
        # B, 128, hidden: layer 2th
        # B, 255, hidden: layer 1th
        # B, 255, hidden: embedding
        offsets = []
        cur = 0
        for t in token_lengths:
            offsets.append((cur, cur + t))
            cur += t
        # offsets: 是vq向量的位置索引，一般是32+64+128+255+255


        # print(offsets, vq_input.shape)
        
        z_q_l, s_l, vq_loss_l, perplexity_l = quantize(vq_input, self.vqmodel)
        # print(z_q_l.shape)
        # print('vq output check: ', torch.isnan(z_q_l).any(), z_q_l.requires_grad)

        z_levels = []
        for (s, e) in offsets:
            z_levels.append(z_q_l[:, s:e])  # z_levels decoder inpute and its KV
        
        decoder_input = z_levels[0]

        coarse_length = decoder_input.shape[1]
        input_dimensions = (coarse_length, 1)

        decoder_outputs = self.decoder(
            hidden_states=decoder_input,          # coarse-level tokens
            input_dimensions=input_dimensions,
            vq_latents=z_levels,               # 每层 KV
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        # print('Decoder Shape: ', decoder_outputs[0].shape)
        hidden_states = decoder_outputs[0].unsqueeze(1).permute(0, 1, 3, 2)
        # print('decoder output check: ', torch.isnan(hidden_states).any(), hidden_states.requires_grad)
        # print(hidden_states.shape) #  b,1,patch, number
        hidden_states = xfold(hidden_states, self.patch_size, self.stride)
        # print(hidden_states.shape)
        ground_truth, reconstruction = self.post_process(hidden_states, time_series, nan_mask)
        # print(reconstruction.shape)
        # print_gpu_mem_usage()
        
        # print(z_q_l.shape, torch.isnan(z_q_l).any(),'VQVAE Output Shape')

        # 计算损失
        # print('reconstruction output check: ', torch.isnan(reconstruction).any(), torch.isnan(ground_truth).sum().item(), torch.isnan(reconstruction).sum().item())
        # print('loss:  ', self.loss(ground_truth, reconstruction), vq_loss_l['loss'])
        losses = {
            'recons_loss_LF': self.loss(ground_truth, reconstruction),
            'recons_loss_HF': None,
            'vq_loss_LF': vq_loss_l,
            'vq_loss_HF': None,
            'total_loss': self.loss(ground_truth, reconstruction) + vq_loss_l['loss'],
        }
        # print(losses)
        perplexity = {
            'total': perplexity_l,
            'perplexity_l': perplexity_l,
            'perplexity_h': None,
        }

        '''# 验证时可视化重构结果
        if self.training and iter_num:
            if iter_num % 1000 == 0:
                with torch.no_grad():
                    self.plot_reconstructions(ground_truth, reconstruction, iter_num)'''

        if not return_reconstruction:
            return losses, perplexity
            # return losses
        else:
            return losses, perplexity, ground_truth, reconstruction
        
    def save_codebook(self, path):
        """
        Input:
            - path: shape [*]; default: required.
        Method:
            - Executes the `Swin.save_codebook` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        torch.save({
            'codebook': self.vqmodel.codebook.detach().cpu(),
            'config': self.config
        }, path)

    def load_codebook(self, path):
        """
        Input:
            - path: shape [*]; default: required.
        Method:
            - Executes the `Swin.load_codebook` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.vqmodel.codebook = checkpoint['codebook'].to(self.vqmodel.codebook.device)

    def plot_reconstructions(self, ground_truth, reconstruction, global_step):
        
        """
        Input:
            - ground_truth: shape [*]; default: required.
            - reconstruction: shape [*]; default: required.
            - global_step: shape [*]; default: required.
        Method:
            - Executes the `Swin.plot_reconstructions` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        with torch.no_grad():

            x = ground_truth
            x_rec = reconstruction.to(x.dtype)
            b = np.random.randint(0, x.shape[0])
            c = np.random.randint(0, x.shape[1])

            alpha = 0.7
            n_rows = 1
            fig, axes = plt.subplots(n_rows, 1, figsize=(4, 2 * n_rows))
            plt.suptitle(f'step-{global_step} | channel idx:{c} \n (blue:GT, orange:reconstructed)')

            torch.cuda.empty_cache()

            # 总体对比
            axes.plot(x[b, c].cpu(), alpha=alpha)
            axes.plot(x_rec[b, c].detach().cpu(), alpha=alpha)
            axes.set_title(r'$x$ (LF+HF)')

            plt.tight_layout()
            
            # 保存图像
            if not os.path.exists('reconstructions_1008'):
                os.makedirs('reconstructions_1008')
            print('savefig')
            plt.savefig(f'reconstructions_1008/step_{global_step}_recon.png')
            plt.close()
            
            
    # ---------------------- 新增方法：获取量化结果 ----------------------
    def encode_to_quantized(self, time_series):
        """
        Input:
            - time_series (torch.Tensor): shape [B, L] or [B, C, L], with C defaulting to 1 when omitted; default: required.
        Method:
            - Runs the encoder-side tokenization path and returns the quantized latent tokens together with the VQ statistics.
        Output:
            - z_q_l (torch.Tensor): shape [B, sum(T_i), hid_dim].
            - s_l (torch.Tensor): shape [B, sum(T_i)], containing codebook indices.
            - vq_loss_l (dict): scalar loss terms produced by the vector quantizer.
            - perplexity_l (torch.Tensor): shape [] or [1], codebook perplexity.
        """
        # 1. 处理输入维度：确保输入为 (B, C, L)（C=self.in_channels，通常为1）
        if len(time_series.shape) == 2:
            # 输入是 (B, L) → 增加通道维度 (B, 1, L)
            time_series = time_series.unsqueeze(1)  # 与原forward逻辑保持一致
        elif len(time_series.shape) != 3:
            raise ValueError(f"输入时序维度必须为 (B, L) 或 (B, C, L)，当前为 {time_series.shape}")

        vq_input, z_q_l, s_l, _ = self.encode_to_patchcode(time_series)
        _, _, vq_loss_l, perplexity_l = quantize(vq_input, self.vqmodel)
        return z_q_l, s_l, vq_loss_l, perplexity_l
    
    def encode_to_patchcode(self, time_series):
        """
        Input:
            - time_series (torch.Tensor): shape [B, L] or [B, C, L], with C defaulting to 1 when omitted; default: required.
        Method:
            - Runs the encoder and vector quantizer without decoding so multiscale token codes can be exported for downstream analysis.
        Output:
            - vq_input (torch.Tensor): shape [B, sum(T_i), hid_dim], pre-quantization encoder tokens.
            - z_q_l (torch.Tensor): shape [B, sum(T_i), hid_dim], quantized tokens.
            - s_l (torch.Tensor): shape [B, sum(T_i)], codebook indices.
            - encoder_outputs[-1]: encoder-side multilevel hidden states returned by `SwinEncoder.forward`.
        """
        if len(time_series.shape) == 2:
            # 输入是 (B, L) → 增加通道维度 (B, 1, L)
            time_series = time_series.unsqueeze(1)  # 与原forward逻辑保持一致
        elif len(time_series.shape) != 3:
            raise ValueError(f"输入时序维度必须为 (B, L) 或 (B, C, L)，当前为 {time_series.shape}")
        
        time_series, nan_mask = self.pre_process(time_series)

        # print(time_series.shape, nan_mask.shape, torch.isnan(time_series).any(),'Transformer Input Shape')

        input_time_series = time_series.permute(0, 2, 1).unsqueeze(1)
        attention_mask = nan_mask.transpose(-1,-2).squeeze(-1)
    
        B, C, L, W= input_time_series.shape
        
        # attention_mask = torch.ones(B, C).to(input_time_series.device).bool()
        # print('input_time_series',input_time_series.shape, attention_mask.shape)

        input_time_series, output_dimensions , mask = self.embeddings(input_time_series, bool_masked_pos=None, interpolate_pos_encoding=False)
        # print(input_time_series)
        # print('embedding check: ', torch.isnan(input_time_series).any(), input_time_series.requires_grad)
        # input_time_series = input_time_series.transpose(-1, -2)

        encoder_outputs =  self.encoder(input_time_series,
            output_dimensions,
            output_attentions = self.output_attentions,
            output_hidden_states = self.output_hidden_states,
            output_hidden_states_before_downsampling = True,
            return_dict=True)

        
        hidden_states = encoder_outputs[1]   # list fine→coarse

        
        hidden_states = list(hidden_states)[::-1]  # 反转成 coarse→fine
        token_lengths = [h.shape[1] for h in hidden_states]
        vq_input = torch.cat(hidden_states, dim=1)
        # print('encoder output check: ', torch.isnan(vq_input).any(), vq_input.requires_grad)
        # hidden_states: layer = 4
        # B, 32, hidden: layer 4th
        # B, 64, hidden: layer 3th
        # B, 128, hidden: layer 2th
        # B, 255, hidden: layer 1th
        # B, 255, hidden: embedding
        offsets = []
        cur = 0
        for t in token_lengths:
            offsets.append((cur, cur + t))
            cur += t
        # offsets: 是vq向量的位置索引，一般是32+64+128+255+255


        # print(offsets, vq_input.shape)
        
        z_q_l, s_l, vq_loss_l, perplexity_l = quantize(vq_input, self.vqmodel)
        # print(z_q_l.shape)
        # print('vq output check: ', torch.isnan(z_q_l).any(), z_q_l.requires_grad)

        z_levels = []
        for (s, e) in offsets:
            z_levels.append(z_q_l[:, s:e])  # z_levels decoder inpute and its KV
        return vq_input, z_q_l, s_l, encoder_outputs[-1]
    

    def mask_L_to_patch(self, mask_L, patch_size, stride):
        """
        Input:
            - mask_L: shape [*]; default: required.
            - patch_size: shape [*]; default: required.
            - stride: shape [*]; default: required.
        Method:
            - Executes the `Swin.mask_L_to_patch` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        B, L = mask_L.shape
        T = (L - patch_size) // stride + 1

        mask_patch = torch.zeros(B, T, dtype=torch.bool, device=mask_L.device)

        for i in range(T):
            s = i * stride
            e = s + patch_size
            mask_patch[:, i] = mask_L[:, s:e].all(dim=-1)

        return mask_patch
    
    def merge_mask(self, mask_fine, T_coarse):
        """
        Input:
            - mask_fine: shape [*]; default: required.
            - T_coarse: shape [*]; default: required.
        Method:
            - Executes the `Swin.merge_mask` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        B, T_fine = mask_fine.shape

        # 每个 coarse token 覆盖的 fine token 数（向上取整）
        group = T_fine / T_coarse

        mask_coarse = torch.zeros(B, T_coarse, dtype=torch.bool, device=mask_fine.device)

        for i in range(T_coarse):
            s = int(round(i * group))
            e = int(round((i + 1) * group))
            e = min(e, T_fine)

            # 只要有 False → False
            mask_coarse[:, i] = mask_fine[:, s:e].all(dim=-1)

        return mask_coarse
    
    def build_multilevel_masks_from_offsets(self, 
    mask_L: torch.Tensor,
    patch_size: int,
    stride: int,
    offsets,
):
        """
        Input:
            - mask_L (torch.Tensor): shape [*]; default: required.
            - patch_size (int): shape []; default: required.
            - stride (int): shape []; default: required.
            - offsets: shape [*]; default: required.
        Method:
            - Executes the `Swin.build_multilevel_masks_from_offsets` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        layer = len(offsets)

        # 各层 T
        Ts = [e - s for (s, e) in offsets]

        # 最细层（最后一个）
        T_patch = Ts[-1]

        # 1. L → finest
        mask = self.mask_L_to_patch(mask_L, patch_size, stride)
        assert mask.shape[1] == T_patch, \
            f"Patch T mismatch: got {mask.shape[1]}, expect {T_patch}"

        level_masks = [None] * layer
        level_masks[-1] = mask

        # 2. 自底向上 merge
        for i in range(layer - 2, -1, -1):
            mask = self.merge_mask(mask, Ts[i])
            level_masks[i] = mask

        return level_masks

    def build_s_l_mask(self,
    mask_L,
    patch_size,
    stride,
    offsets,
):
        """
        Input:
            - mask_L: shape [*]; default: required.
            - patch_size: shape [*]; default: required.
            - stride: shape [*]; default: required.
            - offsets: shape [*]; default: required.
        Method:
            - Executes the `Swin.build_s_l_mask` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        B = mask_L.shape[0]
        total_T = offsets[-1][1]

        s_l_mask = torch.zeros(B, total_T, dtype=torch.bool, device=mask_L.device)

        level_masks = self.build_multilevel_masks_from_offsets(
            mask_L,
            patch_size,
            stride,
            offsets,
        )

        for (s, e), m in zip(offsets, level_masks):
            assert e - s == m.shape[1]
            s_l_mask[:, s:e] = m

        return s_l_mask
    

    
    def single_masked_mae(self, prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
        """
        Input:
            - prediction (torch.Tensor): shape [*]; default: required.
            - target (torch.Tensor): shape [*]; default: required.
            - null_val (float): shape []; default: np.nan.
        Method:
            - Executes the `Swin.single_masked_mae` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """

        if np.isnan(null_val):
            mask = ~torch.isnan(target)
        else:
            eps = 5e-5
            mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

        mask = mask.float()
        mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
        mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

        loss = torch.abs(prediction - target)
        loss = loss * mask  # Apply the mask to the loss
        loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

        return torch.mean(torch.mean(loss, dim=-1), dim=-1)

    def single_masked_mse(self, prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
        """
        Input:
            - prediction (torch.Tensor): shape [*]; default: required.
            - target (torch.Tensor): shape [*]; default: required.
            - null_val (float): shape []; default: np.nan.
        Method:
            - Executes the `Swin.single_masked_mse` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """

        if np.isnan(null_val):
            mask = ~torch.isnan(target)
        else:
            eps = 5e-5
            mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

        mask = mask.float()
        mask /= torch.mean(mask)  # Normalize mask to maintain unbiased MSE calculation
        mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

        loss = (prediction - target) ** 2  # Compute squared error
        loss *= mask  # Apply mask to the loss
        loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

        return torch.mean(torch.mean(loss,dim=-1), dim=-1)  # Return the mean of the masked loss
    

    def inference(self, time_series, return_reconstruction=False, iter_num=None):
        # torch.autograd.set_detect_anomaly(True)
        # print(time_series.device)
        """
        Input:
            - time_series (torch.Tensor): shape [B, L] or [B, C, L], with C defaulting to 1 when omitted; default: required.
            - return_reconstruction (bool): shape []; default: False.
            - iter_num (int | None): shape []; default: None.
        Method:
            - Runs the full encode-quantize-decode path and additionally returns codebook indices plus their multilevel validity masks for analysis.
        Output:
            - when `return_reconstruction=False`: `(s_l, codebook_mask, losses, perplexity)`.
            - when `return_reconstruction=True`: `(s_l, codebook_mask, losses, perplexity, ground_truth, reconstruction)`.
        """
        if len(time_series.shape) == 2:
            time_series = time_series.unsqueeze(1)
        time_series, nan_mask = self.pre_process(time_series)

        # print(time_series.shape, nan_mask.shape, torch.isnan(time_series).any(),'Transformer Input Shape')

        input_time_series = time_series.permute(0, 2, 1).unsqueeze(1)
        attention_mask = nan_mask.transpose(-1,-2).squeeze(-1)
    
        B, C, L, W= input_time_series.shape
        
        # attention_mask = torch.ones(B, C).to(input_time_series.device).bool()
        # print('input_time_series',input_time_series.shape, attention_mask.shape)

        input_time_series, output_dimensions , mask = self.embeddings(input_time_series, bool_masked_pos=None, interpolate_pos_encoding=False)
        # print(input_time_series)
        # print('embedding check: ', torch.isnan(input_time_series).any(), input_time_series.requires_grad)
        # input_time_series = input_time_series.transpose(-1, -2)

        encoder_outputs =  self.encoder(input_time_series,
            output_dimensions,
            output_attentions = self.output_attentions,
            output_hidden_states = self.output_hidden_states,
            output_hidden_states_before_downsampling = True,
            return_dict=True)

        hidden_states = encoder_outputs[1]   # list fine→coarse

        
        hidden_states = list(hidden_states)[::-1]  # 反转成 coarse→fine
        token_lengths = [h.shape[1] for h in hidden_states]
        vq_input = torch.cat(hidden_states, dim=1)
        # print('encoder output check: ', torch.isnan(vq_input).any(), vq_input.requires_grad)
        # hidden_states: layer = 4
        # B, 32, hidden: layer 4th
        # B, 64, hidden: layer 3th
        # B, 128, hidden: layer 2th
        # B, 255, hidden: layer 1th
        # B, 255, hidden: embedding
        offsets = []
        cur = 0
        for t in token_lengths:
            offsets.append((cur, cur + t))
            cur += t
        # offsets: 是vq向量的位置索引，一般是32+64+128+255+255


        # print(offsets, vq_input.shape)
        
        z_q_l, s_l, vq_loss_l, perplexity_l = quantize(vq_input, self.vqmodel)
        # print(z_q_l.shape)
        # print('vq output check: ', torch.isnan(z_q_l).any(), z_q_l.requires_grad)

        z_levels = []
        for (s, e) in offsets:
            # print(s, e)
            z_levels.append(z_q_l[:, s:e])  # z_levels decoder inpute and its KV
        
        decoder_input = z_levels[0]

        coarse_length = decoder_input.shape[1]
        input_dimensions = (coarse_length, 1)

        decoder_outputs = self.decoder(
            hidden_states=decoder_input,          # coarse-level tokens
            input_dimensions=input_dimensions,
            vq_latents=z_levels,               # 每层 KV
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        # print('Decoder Shape: ', decoder_outputs[0].shape)
        hidden_states = decoder_outputs[0].unsqueeze(1).permute(0, 1, 3, 2)
        # print('decoder output check: ', torch.isnan(hidden_states).any(), hidden_states.requires_grad)
        # print(hidden_states.shape) #  b,1,patch, number
        hidden_states = xfold(hidden_states, self.patch_size, self.stride)
        # print(hidden_states.shape)
        ground_truth, reconstruction = self.post_process(hidden_states, time_series, nan_mask)
        # print(reconstruction.shape)
        # print_gpu_mem_usage()
        
        # print(z_q_l.shape, torch.isnan(z_q_l).any(),'VQVAE Output Shape')

        # 计算损失
        # print('reconstruction output check: ', torch.isnan(reconstruction).any(), torch.isnan(ground_truth).sum().item(), torch.isnan(reconstruction).sum().item())
        # print('loss:  ', self.loss(ground_truth, reconstruction), vq_loss_l['loss'])
        losses = {
            'recons_loss_LF': self.loss(ground_truth, reconstruction),
            'recons_loss_HF': None,
            'vq_loss_LF': vq_loss_l,
            'vq_loss_HF': None,
            'total_loss': self.loss(ground_truth, reconstruction) + vq_loss_l['loss'],
            'recons_batch': [self.single_masked_mse(prediction=reconstruction, target=ground_truth), self.single_masked_mae(prediction=reconstruction, target=ground_truth)]
        }
        # print(losses)
        perplexity = {
            'total': perplexity_l,
            'perplexity_l': perplexity_l,
            'perplexity_h': None,
        }

        '''# 验证时可视化重构结果
        if self.training and iter_num:
            if iter_num % 1000 == 0:
                with torch.no_grad():
                    self.plot_reconstructions(ground_truth, reconstruction, iter_num)'''
        
        codebook_mask = self.build_s_l_mask(attention_mask, self.patch_size, self.stride, offsets)

        if not return_reconstruction:
            
            return s_l.cpu(),  codebook_mask.cpu(), losses, perplexity
            # return losses
        else:
            # print(s_l.shape, codebook_mask)
            return s_l.cpu(),  codebook_mask.cpu(), losses, perplexity, ground_truth.cpu(), reconstruction.cpu()






