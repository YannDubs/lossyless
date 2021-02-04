import logging
import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data

__all__ = ["ClassicalCompressor"]

logger = logging.getLogger(__name__)


class ClassicalCompressor(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # TODO
        # self.compressor = ...

    @auto_move_data  # move data on correct device for inference
    def forward(self, x, **kwargs):
        """Represents the data `x`.

        Parameters
        ----------
        X : torch.Tensor of shape=[batch_size, *data.shape]
            Data to compress.

        kwargs : 
            Placeholder.

        Returns
        -------
        if is_features:
            z : torch.Tensor of shape=[batch_size, z_dim]
                Represented data.
        else:
            X_hat : torch.Tensor of shape=[batch_size,  *data.shape]
                Reconstructed data.
        """
        return x

    # def step(self, batch):
    #     cfgl = self.hparams.loss
    #     x, (_, aux_target) = batch
    #     n_z = cfgl.n_z_samples

    #     # batch shape: [batch_size] ; event shape: [z_dim]
    #     p_Zlx = self.p_ZlX(x)

    #     # shape: [n_z, batch_size, z_dim]
    #     z = p_Zlx.rsample([n_z])

    #     # z_hat. shape: [n_z, batch_size, z_dim]
    #     z_hat, rates, r_logs, r_other = self.rate_estimator(z, p_Zlx)

    #     distortions, d_logs, d_other = self.distortion_estimator(
    #         z_hat, aux_target, p_Zlx
    #     )

    #     loss, logs, other = self.loss(rates, distortions)

    #     # to log (dict)
    #     logs.update(r_logs)
    #     logs.update(d_logs)
    #     logs.update(dict(zmin=z.min(), zmax=z.max(), zmean=z.mean()))

    #     # any additional information that can be useful (dict)
    #     other.update(r_other)
    #     other.update(d_other)
    #     other["X"] = x[0].detach().cpu()
    #     other.update(dict(z=z.detach()))

    #     return loss, logs, other

    # def loss(self, rates, distortions):
    #     n_z = rates.size(0)
    #     cfg = self.hparams
    #     beta = cfg.loss.beta * cfg.rate.factor_beta * cfg.distortion.factor_beta

    #     # loose_loss for plotting. shape: []
    #     loose_loss = (distortions + beta * rates).mean()

    #     # tightens bound using IWAE: log 1/k sum exp(loss). shape: [batch_size]
    #     if n_z > 1:
    #         rates = torch.logsumexp(rates, 0) - math.log(n_z)
    #         distortions = torch.logsumexp(distortions, 0) - math.log(n_z)
    #     else:
    #         distortions = distortions.squeeze(0)
    #         rates = rates.squeeze(0)

    #     # E_x[...]. shape: shape: []
    #     rate = rates.mean(0)
    #     distortion = distortions.mean(0)
    #     loss = distortion + beta * rate

    #     logs = dict(
    #         loose_loss=loose_loss / math.log(BASE_LOG),
    #         loss=loss / math.log(BASE_LOG),
    #         rate=rate / math.log(BASE_LOG),
    #         distortion=distortion / math.log(BASE_LOG),
    #     )
    #     other = dict()

    #     return loss, logs, other

    # def test_step(self, batch, batch_idx):
    #     loss, logs, _ = self.step(batch)
    #     _, online_logs = self.online_evaluator(batch, self)
    #     logs.update(online_logs)
    #     self.log_dict(
    #         {f"test/{k}": v for k, v in logs.items()}, on_epoch=True, on_step=False
    #     )
    #     return loss
