import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import numpy as np
import collections.abc


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


class SwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        """
        Input:
            - config: shape object; default: required.
            - use_mask_token: shape [*]; default: False.
        Method:
            - Executes the `SwinEmbeddings.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()

        self.patch_embeddings = SwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = (config.context_length, 0)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
        else:
            self.position_embeddings = None

        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.config = config

    # Copied from transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Input:
            - embeddings (torch.Tensor): shape [*]; default: required.
            - height (int): shape []; default: required.
            - width (int): shape []; default: required.
        Method:
            - Executes the `SwinEmbeddings.interpolate_pos_encoding` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor],
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> tuple[torch.Tensor]:
        
        """
        Input:
            - pixel_values (torch.FloatTensor): shape [B, C, H, W], where this project typically uses [B, 1, L, 1]; default: required.
            - bool_masked_pos (torch.BoolTensor | None): shape [B, num_patches] or None; default: None.
            - interpolate_pos_encoding (bool): shape []; default: False.
        Method:
            - Projects raw inputs into patch tokens, optionally replaces masked positions with a learned mask token, and adds positional embeddings.
        Output:
            - embeddings (torch.Tensor): shape [B, num_patches, embed_dim].
            - output_dimensions (tuple[int, int]): shape [2], containing patch-grid height and width.
            - mask (torch.Tensor | None): shape [B, num_patches] or None.
        """

        _, num_channels, height, width = pixel_values.shape
        # print('fake_image/timeseries shape: ', pixel_values.shape, num_channels, height, width, bool_masked_pos.shape)
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            bool_masked_pos = bool_masked_pos.unfold(dimension=-1, size=self.patch_size, step=self.stride).squeeze().all(dim=-1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            # print('mask_shape: ', embeddings.shape, mask[0])
            embeddings = embeddings * mask + mask_tokens * (1.0 - mask)
        else:
            mask = None

        if self.position_embeddings is not None:
            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                # print(embeddings.shape, self.position_embeddings.shape, 'xxxxxxxxxxxxxxxxx')
                embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings, output_dimensions, mask


class SwinPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        """
        Input:
            - config: shape object; default: required.
        Method:
            - Executes the `SwinPatchEmbeddings.__init__` logic using the provided inputs and current module state.
        Output:
            - return: None. The module stores the initialized submodules and hyperparameters in `self`.
        """
        super().__init__()
        image_size, patch_size = config.in_channels, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = config.context_length#(image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (config.context_length, 0)

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=config.stride)

    def maybe_pad(self, pixel_values, height, width):
        """
        Input:
            - pixel_values: shape [*]; default: required.
            - height: shape []; default: required.
            - width: shape []; default: required.
        Method:
            - Executes the `SwinPatchEmbeddings.maybe_pad` logic using the provided inputs and current module state.
        Output:
            - return: value(s) produced by this helper; exact shape follows the implementation below.
        """
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> tuple[torch.Tensor, tuple[int]]:
        """
        Input:
            - pixel_values (torch.FloatTensor | None): shape [B, C, H, W]; default: required.
        Method:
            - Pads the raw series when needed and applies the patch projection convolution to obtain patch tokens.
        Output:
            - embeddings (torch.Tensor): shape [B, num_patches, embed_dim].
            - output_dimensions (tuple[int, int]): shape [2], containing padded patch-grid height and width.
        """
        _, num_channels, height, width = pixel_values.shape
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, output_dimensions
