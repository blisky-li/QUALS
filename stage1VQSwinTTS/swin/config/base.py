from easydict import EasyDict


in_channels = 4096
input_length = 1
hid_dim = 256
patch_size = 32
stride = 16
depth = [2, 2, 2, 2]

dropout = 0.1
attention_dropout = 0.1
activation_dropout = 0.1

encoder_layerdrop = 0.1
decoder_layerdrop = 0.1
encoder_layers = 4
decoder_layers = 4

encoder_attention_heads = 8
decoder_attention_heads = 8

encoder_ffn_dim = 64
decoder_ffn_dim = 64

activation_function = "gelu"
init_std = 0.1


SWINTTS_BASE_CONFIG = EasyDict(
    {
        "in_channels": in_channels,
        "feature_size": patch_size,
        "num_channels": input_length,
        "context_length": (in_channels - patch_size) // stride + 1,
        "perdict_length": input_length,
        "window_size": 17,
        "embed_dim": hid_dim,
        "depths": depth,
        "num_heads": [2, 4, 8, 16],
        "num_layers": len(depth),
        "patch_size": patch_size,
        "stride": stride,
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.1,
        "hid_dim": hid_dim,
        "d_model": hid_dim,
        "layer_norm_eps": 1e-5,
        "dropout": dropout,
        "hidden_dropout_prob": dropout,
        "attention_dropout": attention_dropout,
        "attention_probs_dropout_prob": attention_dropout,
        "activation_dropout": activation_dropout,
        "qkv_bias": True,
        "use_absolute_embeddings": True,
        "encoder_layerdrop": encoder_layerdrop,
        "decoder_layerdrop": decoder_layerdrop,
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "encoder_attention_heads": encoder_attention_heads,
        "decoder_attention_heads": decoder_attention_heads,
        "encoder_ffn_dim": encoder_ffn_dim,
        "decoder_ffn_dim": decoder_ffn_dim,
        "init_std": init_std,
        "hidden_act": activation_function,
        "output_hidden_states": False,
        "output_attentions": False,
        "vq_vae": EasyDict(
            {
                "commitment_weight": 0.95,
                "decay": 0.97,
                "codebook_size": 2048,
                "codebook_dim": 32,
                "kmeans_init": False,
                "kmeans_iters": 200,
                "sync_codebook": True,
                "use_cosine_sim": True,
                "rotation_trick": True,
            }
        ),
    }
)
