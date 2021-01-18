import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
import torch
import logging
import einops
import math
from .rates import get_rate_estimator
from .architectures import get_Architecture, MLP
from .distributions import CondDist, get_marginalDist
from .helpers import get_lr_scheduler, BASE_LOG, orderedset
from .distortions import get_distortion_estimator
from .predictors import OnlineEvaluator


__all__ = ["CompressionModule"]

logger = logging.getLogger(__name__)


class CompressionModule(pl.LightningModule):
    """Main network for compression."""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.p_ZlX = self.get_encoder()  # p_{Z | X}
        self.rate_estimator = self.get_rate_estimator()
        self.distortion_estimator = self.get_distortion_estimator()
        self.online_evaluator = self.get_online_evaluator()

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

    def get_rate_estimator(self):
        """Return the correct rate estimator. Contains the prior and the coder."""
        cfg_rate = self.hparams.rate
        return get_rate_estimator(
            cfg_rate.name,
            z_dim=self.hparams.encoder.z_dim,
            p_ZlX=self.p_ZlX,
            n_z_samples=self.hparams.loss.n_z_samples,
            **cfg_rate.kwargs,
        )

    def get_distortion_estimator(self):
        """Return the correct distortion estimator. Contains the decoder."""
        cfg_dist = self.hparams.distortion
        return get_distortion_estimator(
            cfg_dist.mode, p_ZlX=self.p_ZlX, **cfg_dist.kwargs
        )

    def get_online_evaluator(self):
        """
        Online evaluation of the representation. replaces pl_bolts.callbacks.SSLOnlineEValuator
        because training as a callbackwas not well support by lightning. E.g. continuing training
        from checkpoint.
        """
        # TODO maybe use same parameters as the actual downstream predictor
        return OnlineEvaluator(
            self.hparams.encoder.z_dim,
            self.hparams.data.target_shape,
            is_classification=self.hparams.data.target_is_clf,
        )

    @auto_move_data  # move data on correct device for inference
    def forward(
        self,
        x,
        is_compress=False,
    ):
        """Represents the data `x`.

        Parameters
        ----------
        X : torch.Tensor of shape=[batch_size, *data.shape]
            Data to represent.

        is_compress : bool, optional
            Whether to return the compressed representation.

        Returns
        -------
        z : torch.Tensor of shape=[batch_size, z_dim]
            Represented data.
        """
        p_Zlx = self.p_ZlX(x)
        z = p_Zlx.rsample([1])

        # shape: [batch_size, z_dim]
        if is_compress:
            z = z.squeeze(0)
            z_hat = self.rate_estimator.compress(z)
        else:
            z_hat, *_ = self.rate_estimator(z, p_Zlx)
            z_hat = z_hat.squeeze(0)

        return z_hat

    def step(self, batch):
        cfgl = self.hparams.loss
        x, (_, aux_target) = batch
        n_z = cfgl.n_z_samples

        # batch shape: [batch_size] ; event shape: [z_dim]
        p_Zlx = self.p_ZlX(x)

        # shape: [n_z, batch_size, z_dim]
        z = p_Zlx.rsample([n_z])

        # z_hat. shape: [n_z, batch_size, z_dim]
        z_hat, rates, r_logs, r_other = self.rate_estimator(z, p_Zlx)

        distortions, d_logs, d_other = self.distortion_estimator(
            z_hat, aux_target, p_Zlx
        )

        loss, logs, other = self.loss(rates, distortions)

        # to log (dict)
        logs.update(r_logs)
        logs.update(d_logs)
        logs.update(dict(zmin=z.min(), zmax=z.max(), zmean=z.mean()))

        # any additional information that can be useful (dict)
        other.update(r_other)
        other.update(d_other)
        other.update(dict(z=z.detach()))

        return loss, logs, other

    def loss(self, rates, distortions):
        n_z = rates.size(0)
        cfg = self.hparams
        beta = cfg.loss.beta * cfg.rate.factor_beta * cfg.distortion.factor_beta

        # loose_loss for plotting. shape: []
        loose_loss = (distortions + beta * rates).mean()

        # tightens bound using IWAE: log 1/k sum exp(loss). shape: [batch_size]
        if n_z > 1:
            rates = torch.logsumexp(rates, 0) - math.log(n_z)
            distortions = torch.logsumexp(distortions, 0) - math.log(n_z)
        else:
            distortions = distortions.squeeze(0)
            rates = rates.squeeze(0)

        # E_x[...]. shape: shape: []
        rate = rates.mean(0)
        distortion = distortions.mean(0)
        loss = distortion + beta * rate

        logs = dict(
            loose_loss=loose_loss / math.log(BASE_LOG),
            loss=loss / math.log(BASE_LOG),
            rate=rate / math.log(BASE_LOG),
            distortion=distortion / math.log(BASE_LOG),
        )
        other = dict()

        return loss, logs, other

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        # MODEL
        if optimizer_idx == 0:
            loss, logs, other = self.step(batch)

            #! waiting for torch lightning #1243
            #! everyting in other should be detached
            self._save = other

        # ONLINE EVALUATOR
        elif optimizer_idx == 1:
            loss, logs = self.online_evaluator(batch, self)

        # CODER
        else:
            # TODO make sure that ok if online evaluator is being run before getting coder_loss
            loss = self.rate_estimator.aux_loss()
            logs = dict(coder_loss=loss)

        self.log_dict({f"train_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs, _ = self.step(batch)
        _, online_logs = self.online_evaluator(batch, self)
        logs.update(online_logs)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs, _ = self.step(batch)
        _, online_logs = self.online_evaluator(batch, self)
        logs.update(online_logs)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return loss

    def on_validation_epoch_start(self):
        """Make sure that you can actually use the coder during eval."""
        self.rate_estimator.update(force=True)

    def yield_parameters(self):
        """Returns an iterator over the model parameters."""
        for m in self.children():
            # slightly different than default because calling .parameters() on each
            # so can overide .parameters in submodules.
            for p in m.parameters():
                yield p

    def parameters(self):
        """Returns an iterator over the model parameters."""
        # return a set to make sure  not counting parameters twice (e.g. P_ZlX can be in multiple child)
        # but keep ordered which is needed for checkpointing
        return orderedset(self.yield_parameters())

    def configure_optimizers(self):

        aux_parameters = orderedset(self.rate_estimator.aux_parameters())
        online_parameters = orderedset(self.online_evaluator.aux_parameters())
        is_optimize_coder = len(aux_parameters) > 0
        epochs = self.hparams.trainer.max_epochs

        optimizers = []
        schedulers = []

        # MODEL OPTIMIZER
        cfgo = self.hparams.optimizer

        optimizer = torch.optim.Adam(
            self.parameters(), lr=cfgo.lr, weight_decay=cfgo.weight_decay
        )
        if cfgo.is_lars:
            optimizer = LARSWrapper(optimizer)
        optimizers += [optimizer]

        scheduler = get_lr_scheduler(optimizer, epochs=epochs, **cfgo.scheduler)
        if scheduler is not None:
            schedulers += [scheduler]

        # ONLINE EVALUATOR
        # do not use scheduler for online eval because input (representation) is changing
        online_optimizer = torch.optim.Adam(online_parameters, lr=1e-4)
        optimizers += [online_optimizer]

        # CODER OPTIMIZER
        if is_optimize_coder:
            cfgoc = self.hparams.optimizer_coder
            optimizer_coder = torch.optim.Adam(aux_parameters, lr=cfgoc.lr)
            optimizers += [optimizer_coder]

            scheduler = get_lr_scheduler(
                optimizer_coder,
                epochs=epochs,
                **cfgoc.scheduler,
            )
            if scheduler is not None:
                schedulers += [scheduler]

        return optimizers, schedulers
