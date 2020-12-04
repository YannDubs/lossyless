import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
import torch
import logging
import einops
from .coders import get_coder
from .architectures import get_Architecture
from .distributions import CondDist, get_marginalDist
from .helpers import get_exponential_decay_gamma, mean
from .losses import get_loss


__all__ = ["CompressionModule"]

logger = logging.getLogger(__name__)


class CompressionModule(pl.LightningModule):
    """Main network for compression."""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.p_ZlX = self.get_encoder()  # p_{Z | X}
        self.q_YlZ = self.get_decoder()  # q_{Y | Z}
        self.coder = self.get_coder()  # contains the prior and the coder
        self.loss = get_loss(self.hparams.loss.mode, **self.hparams.loss.kwargs)

    def get_encoder(self):
        """Return encoder: a mapping to a torch.Distribution (conditional distribution)."""
        cfg_enc = self.hparams.encoder
        return CondDist(
            self.hparams.data.shape,
            cfg_enc.z_dim,
            family=cfg_enc.fam,
            Architecture=get_Architecture(cfg_enc.arch, **cfg_enc.arch_kwargs),
            **cfg_enc.fam_kwargs,
        )

    def get_decoder(self):
        """Return decoder: a predictor to a torch.Tensor (expected value of conditional distribution)."""
        cfg_dec = self.hparams.decoder
        Decoder = get_Architecture(cfg_dec.arch, **cfg_dec.arch_kwargs)
        return Decoder(self.hparams.encoder.z_dim, cfg_dec.out_shape)

    def get_coder(self):
        """Return the correct entropy coder."""
        cfg_cod = self.hparams.coder

        return get_coder(
            cfg_cod.name,
            z_dim=self.hparams.encoder.z_dim,
            p_ZlX=self.p_ZlX,
            **cfg_cod.kwargs,
        )

    @auto_move_data  # move data on correct device for inference
    def forward(
        self, x, is_compress=False,
    ):
        """Represents the data `x`.
        
        Parameters
        ----------
        X : torch.tensor of shape=(*, *data.shape)
            Data to represent.

        is_compress : bool, optional
            Whether to return the compressed representation. 

        Returns
        -------
        z : torch.tensor of shape=(*, z_dim)
            Represented data.
        """
        p_Zlx = self.p_ZlX(x)
        z = p_Zlx.rsample([1]).squeeze(0)

        if is_compress:
            z = self.coder.compress(z)

        return z

    def step(self, batch):

        cfgl = self.hparams.loss
        x, (_, aux_target) = batch
        n_z = cfgl.n_z_samples

        # batch shape: [batch_size] ; event shape: [z_dim]
        p_Zlx = self.p_ZlX(x)

        # shape: [n_z, batch_size, z_dim]
        z = p_Zlx.rsample([n_z])

        # z_hat shape: [n_z, batch_size, z_dim]
        z_hat, coding_loss, coding_logs = self.coder(z, p_Zlx)
        flat_z_hat = einops.rearrange(z_hat, "n_z b d -> (n_z b) d")

        # Y_hat. shape: [n_z_samples, batch_size, *y_shape]
        Y_hat = self.q_YlZ(flat_z_hat)  # Y_hat is suff statistic of q_{Y|z}
        Y_hat = einops.rearrange(Y_hat, "(n_z b) ... -> n_z b ...", n_z=n_z)

        loss, logs = self.loss(Y_hat, aux_target, coding_loss)
        logs.update(coding_logs)
        logs.update(dict(zmin=z.min(), zmax=z.max(), zmean=z.mean()))  # DEV

        other = dict(y_hat=Y_hat.mean(0).detach(), z=z.detach())
        return loss, other, logs

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        if optimizer_idx == 0:
            loss, other, logs = self.step(batch)
            self.log_dict({f"train_{k}": v for k, v in logs.items()})

            #! waiting for torch lighning #1243
            self._save = dict(
                y_hat=other["y_hat"][0].cpu(), target=batch[1][1][0].cpu()
            )

            return loss

        else:  # only if self.coder.is_need_optimizer
            coder_loss = self.coder.loss()
            # updates the coder
            return coder_loss

    def validation_step(self, batch, batch_idx):
        loss, other, logs = self.step(batch)

        if self.coder.is_can_compress:
            z = other["z"]
            z = einops.rearrange(z, "n_z b d -> (n_z b) d", n_z=z.size(0))
            z_compressed = self.coder.compress(z)
            logs["n_bits"] = mean([len(zi) for zi in z_compressed]) * 8  # len in bytes

        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, logs = self.step(batch)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return loss

    def on_validation_epoch_start(self):
        """Make sure that you can actually use the coder during eval."""
        self.coder.update(force=True)

    def configure_optimizers(self):

        # model
        cfgo = self.hparams.optimizer

        if self.coder.is_need_optimizer:
            # use separate optimizer for coder and model
            params = list(self.named_parameters())
            params_model = [p for n, p in params if "coder" not in n]
        else:
            params_model = self.parameters()

        optimizer = torch.optim.Adam(params_model, lr=cfgo.lr)
        gamma = get_exponential_decay_gamma(
            cfgo.scheduling_factor, self.hparams.trainer.max_epochs,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

        optimizers = [optimizer]
        schedulers = [scheduler]

        # if coder needs parameters + optimizer
        if self.coder.is_need_optimizer:
            cfgoc = self.hparams.optimizer_coder
            optimizer_coder = torch.optim.Adam(self.coder.parameters(), lr=cfgoc.lr)
            gamma_coder = get_exponential_decay_gamma(
                cfgoc.scheduling_factor, self.hparams.trainer.max_epochs,
            )
            scheduler_coder = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma_coder
            )
            optimizers += [optimizer_coder]
            schedulers += [scheduler_coder]

        return optimizers, schedulers
