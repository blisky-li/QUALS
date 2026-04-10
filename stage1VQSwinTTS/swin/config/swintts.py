# 采样概率变化

import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from .base import SWINTTS_BASE_CONFIG
from ..arch import Swin
from ..data import BLASTDataset
from ..runner import VQVAETrainRunner

def fake_loss(loss):
    return loss

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
# Model architecture and parameters

MODEL_ARCH = Swin
DATA_NAME = "Synth"

NUM_ITERATIONS = 200000 # 总轮数
VAL_ITERATION_INTERVAL = 2000# 5_000 # 每VAL_ITERATION_INTERVAL执行一次验证

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'Transformer_VQVAE'
CFG.GPU_NUM = 4
# Runner
CFG.RUNNER = VQVAETrainRunner

############################## Model Configuration ################################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = {
                "config": SWINTTS_BASE_CONFIG
                }
CFG.MODEL.DTYPE = 'bfloat16'
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({})

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.COMPILE_MODEL = False
CFG.TRAIN.NUM_ITERATIONS = NUM_ITERATIONS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_ITERATIONS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = VAL_ITERATION_INTERVAL * 1 # 保存策略，每VAL_ITERATION_INTERVAL * 5保存一次模型
CFG.TRAIN.LOSS = fake_loss
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 1e-4,
    "betas": (0.9, 0.95),
    "fused": True,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineWarmup"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'num_warmup_steps': int(NUM_ITERATIONS / 100 * 1), # 10%的warmup启动比例
    'num_training_steps': NUM_ITERATIONS,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 1.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 512
CFG.TRAIN.DATA.SHUFFLE = True # has to be False
CFG.TRAIN.DATA.NUM_WORKERS = 8
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.DATA.PREFETCH = True
CFG.TRAIN.GRAD_ACCUMULATION_STEPS = 1

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = VAL_ITERATION_INTERVAL
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()
# Evaluation parameters
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = BLASTDataset
CFG.DATASET.PARAM = EasyDict({
    'num_valid_samples': 1000
})
