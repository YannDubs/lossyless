import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
import torch
import logging
import einops
from .architectures import get_Architecture
from .distributions import CondDist, get_marginalDist
from .helpers import get_exponential_decay_gamma
from .losses import get_loss


try:
    import wandb
except ImportError:
    pass

__all__ = ["CompressionModule"]

logger = logging.getLogger(__name__)


class CompressionModule(pl.LightningModule):
    """Main network for compression."""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.p_ZlX = self.get_encoder()  # p_{Z | X}
        self.q_YlZ = self.get_decoder()  # q_{Y | Z}
        self.q_Z = self.get_prior()  # q_{Z}
        self.entropy_coder = self.get_entropy_coder()
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

    def get_prior(self):
        """Return prior: a torch.Distribution."""
        return get_marginalDist(
            self.hparams.prior.fam,
            cond_dist=self.p_ZlX,
            **self.hparams.prior.fam_kwargs,
        )

    def get_entropy_coder(self):
        """Return the correct entropy coder."""
        pass

    @auto_move_data  # move data on correct device for inference
    def forward(self, x, is_entropy_code=False):  # TODO entropy code
        """Represents the data `x`.
        
        Parameters
        ----------
        X : torch.tensor of shape=(*, *data.shape)
            Data to represent.

        is_entropy_code :
            Whether to return the represented data after entropy coding (i.e. reconstructed). 
        """
        p_Zlx = self.p_ZlX(x)
        return p_Zlx.rsample([])

    def step(self, batch):

        cfgl = self.hparams.loss
        x, (_, aux_target) = batch

        # batch shape: [batch_size] ; event shape: [z_dim]
        p_Zlx = self.p_ZlX(x)

        # batch shape: [l] ; event shape: [z_dim]
        q_Z = self.q_Z()  # conditioning on nothing

        # shape: [n_z_samples, batch_size, z_dim]
        z_samples = p_Zlx.rsample([cfgl.n_z_samples])

        # breakpoint()

        # shape: [n_z_samples, batch_size, *y_dims]
        flat_z_samples = einops.rearrange(z_samples, "z b ... -> (z b) ...")
        Y_hat = self.q_YlZ(flat_z_samples)  # Y_hat is suff statistic of q_{Y|z}
        Y_hat = einops.rearrange(Y_hat, "(z b) ... -> z b ...", z=cfgl.n_z_samples)

        # breakpoint()

        loss, logs = self.loss(
            Y_hat, aux_target, p_Zlx, q_Z, z_samples, self.entropy_coder
        )

        return loss, Y_hat.mean(0), logs

    def training_step(self, batch, _):
        loss, Y_hat, logs = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})

        # waiting for torch lighning #1243
        hiddens = dict(y_hat=Y_hat[0].detach(), target=batch[1][1][0])

        return dict(loss=loss, hiddens=hiddens)

    def validation_step(self, batch, _):
        loss, Y_hat, logs = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, _):
        loss, logs = self.step(batch)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        gamma = get_exponential_decay_gamma(
            self.hparams.optimizer.scheduling_factor, self.hparams.trainer.max_epochs
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        return [optimizer], [scheduler]
