import torch
import torch.nn.functional as F
from basicts.runners.base_utsf_runner import BaseUniversalTimeSeriesForecastingRunner
from ..utils import linear_warmup_cosine_annealingLR

class VQVAETrainRunner(BaseUniversalTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        
        self.config = cfg

    def forward(self, data: dict,** kwargs) -> dict:
        """前向传播，返回损失组件"""
        time_series = data['time_series']
        return_reconstruction = False
        time_series = self.to_running_device(time_series)
        if return_reconstruction == False:
            losses, perplexity = self.model(time_series,return_reconstruction=return_reconstruction, iter_num = kwargs.get('iter_num'))
            # losses = self.model(time_series,return_reconstruction=return_reconstruction, iter_num = kwargs.get('iter_num'))
        # 调用模型获取损失
        else:
            losses, perplexity, ground_truth, reconstruction = self.model(time_series, return_reconstruction=return_reconstruction, iter_num = kwargs.get('iter_num'))
            # losses, ground_truth, reconstruction = self.model(time_series, return_reconstruction=return_reconstruction, iter_num = kwargs.get('iter_num'))
        
        return {'loss': losses['total_loss']}

    