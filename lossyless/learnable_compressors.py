import logging
import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.core.decorators import auto_move_data

from .architectures import get_Architecture
from .distortions import get_distortion_estimator
from .distributions import CondDist, Deterministic
from .helpers import (BASE_LOG, Annealer, OrderedSet, Timer,
                      append_optimizer_scheduler_)
from .predictors import OnlineEvaluator
from .rates import get_rate_estimator

__all__ = ["LearnableCompressor"]

logger = logging.getLogger(__name__)


class LearnableCompressor(pl.LightningModule):
    """Main network for learning a neural compression."""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.p_ZlX = self.get_encoder()
        self.rate_estimator = self.get_rate_estimator()
        self.distortion_estimator = self.get_distortion_estimator()

        if self.hparams.evaluation.featurizer.is_online:
            self.online_evaluator = self.get_online_evaluator()

        #! careful this is actually 1/beta from the paper (oops)
        final_beta = self.final_beta_labda[0]
        self.beta_annealer = Annealer(
            final_beta * 1e-5,  # don't use 0 in case geometric
            final_beta,
            n_steps_anneal=math.ceil(
                1 / 10 * self.hparams.data.max_steps
            ),  # arbitrarily 1/10th
            mode=self.hparams.featurizer.loss.beta_anneal,
        )

        # governs how the compressor acts when calling it directly
        self.is_features = self.hparams.featurizer.is_features
        self.out_shape = (
            (self.hparams.encoder.z_dim,)
            if self.is_features
            else self.hparams.data.shape
        )

        self.stage = self.hparams.stage  # allow changing to stages

    def get_encoder(self):
        """Return encoder: a mapping to a torch.Distribution (conditional distribution)."""
        # TODO should remove explicit parameters and only call **cfg_enc.kwargs
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
        rate_estimator = get_rate_estimator(
            cfg_rate.mode, p_ZlX=self.p_ZlX, **cfg_rate.kwargs,
        )
        # ensure that pickable before DataDistributed
        rate_estimator.make_pickable_()
        return rate_estimator

    def get_distortion_estimator(self):
        """Return the correct distortion estimator. Contains the decoder."""
        cfg_dist = self.hparams.distortion
        return get_distortion_estimator(cfg_dist.mode, **cfg_dist.kwargs)

    def get_online_evaluator(self):
        """
        Online evaluation of the representation. replaces pl_bolts.callbacks.SSLOnlineEValuator
        because training as a callbackwas not well support by lightning. E.g. continuing training
        from checkpoint.
        """
        # TODO should remove explicit parameters and only call **cfgoe.kwargs
        cfgoe = self.hparams.online_evaluator
        return OnlineEvaluator(
            self.hparams.encoder.z_dim,
            self.hparams.data.target_shape,
            Architecture=get_Architecture(cfgoe.arch, **cfgoe.arch_kwargs),
            **cfgoe.loss_kwargs,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Predict function, this will represent the data and also return the correct label.
        Which is useful in case you want to create a featurized dataset.
        """
        x, y = batch

        if isinstance(y, (tuple, list)):
            y = y[0]  # only return the real label assumed to be first

        x_hat = self(x)
        return x_hat, y

    def predict(self, *args, **kwargs):  # TODO remove in newer version of lightning
        return self.predict_step(*args, **kwargs)

    @auto_move_data  # move data on correct device for inference
    def forward(self, x, is_compress=False, is_features=None):
        """Represents the data `x`.

        Parameters
        ----------
        x : torch.Tensor of shape=[batch_size, *data.shape]
            Data to represent.

        is_compress : bool, optional
            Whether to perform actual compression. If not will simply apply the discretization as
            if we had compressed.

        is_features : bool or None, optional
            Whether to return the features / codes / representation or the reconstructed example.
            Recontructed image only works for distortions that predict reconctructions (e.g. VAE),
            If `None` uses the default from `hparams`.

        Returns
        -------
        if is_features:
            z : torch.Tensor of shape=[batch_size, z_dim]
                Represented data.
        else:
            X_hat : torch.Tensor of shape=[batch_size,  *data.shape]
                Reconstructed data. If image it's the unormalized image in [0,1].
        """

        if is_features is None:
            is_features = self.is_features

        p_Zlx = self.p_ZlX(x)
        z = p_Zlx.rsample()

        # shape: [batch_size, z_dim]
        if is_compress:
            z_hat = self.rate_estimator.compress(z, self)
        else:
            z_hat, *_ = self.rate_estimator(z, p_Zlx, self)

        if is_features:
            out = z_hat
        else:
            # only if direct distortion
            x_hat = self.distortion_estimator.q_YlZ(z_hat)
            if self.distortion_estimator.is_img_out:
                # if working with an image put it back to [0,1]
                x_hat = torch.sigmoid(x_hat)
            out = x_hat

        return out

    def step(self, batch, is_rate_only=False):
        x, targets = batch

        with Timer() as encoder_timer:
            # batch shape: [batch_size] ; event shape: [z_dim]
            p_Zlx = self.p_ZlX(x)

        # shape: [batch_size, z_dim]
        z = p_Zlx.rsample()

        # z_hat. shape: [batch_size, z_dim]
        z_hat, rates, r_logs, r_other = self.rate_estimator(z, p_Zlx, self)

        if "n_bits" in r_logs:
            batch_size = x.size(0)
            r_logs["encoder_time"] = encoder_timer.duration / batch_size
            r_logs["sender_time"] = r_logs["encoder_time"] + r_logs["compress_time"]

            if self.hparams.data.mode == "image":
                _, __, height, width = x.shape
                r_logs["bpp"] = r_logs["n_bits"] / (height * width)

        if is_rate_only:
            r_logs["rate"] = rates.mean() / math.log(BASE_LOG)
            return rates, r_logs, r_other

        _, aux_target = targets
        distortions, d_logs, d_other = self.distortion_estimator(
            z_hat, aux_target, p_Zlx, self
        )

        loss, logs, other = self.loss(rates, distortions)

        # to log (dict)
        logs.update(r_logs)
        logs.update(d_logs)
        logs.update(dict(zmin=z_hat.min(), zmax=z_hat.max(), zmean=z_hat.mean()))

        # any additional information that can be useful (dict)
        other.update(r_other)
        other.update(d_other)
        other["X"] = x[0].detach().cpu()
        other.update(dict(z=z.detach()))

        return loss, logs, other

    @property
    def final_beta_labda(self):
        """Return the final beta to use."""
        cfg = self.hparams

        labda = 1 / cfg.distortion.factor_beta
        beta = cfg.featurizer.loss.beta * cfg.rate.factor_beta

        return beta, labda

    def loss(self, rates, distortions):
        # multiplication by small or large beta might cause issues in float16
        rates = rates.float()
        distortions = distortions.float()

        curr_beta = self.beta_annealer(n_update_calls=self.global_step)
        final_beta, labda = self.final_beta_labda

        # loose_loss for plotting. shape: []
        loose_loss = (labda * distortions + final_beta * rates).mean().detach()

        # E_x[...]. shape: shape: []
        rate = rates.mean(0)
        distortion = distortions.mean(0)

        # use actual (annealed) beta for the gradients, but still want the loss to be in terms of
        # final beta for plotting and checkpointing => use trick
        beta_rate = curr_beta * rate  # actual gradients
        beta_rate = beta_rate - beta_rate.detach() + (final_beta * rate.detach())

        loss = labda * distortion + beta_rate

        logs = dict(
            loose_loss=loose_loss / math.log(BASE_LOG),
            loss=loss / math.log(BASE_LOG),
            rate=rate / math.log(BASE_LOG),
            distortion=distortion / math.log(BASE_LOG),
            beta=curr_beta,
        )
        # if both are entropies this will say how good the model is
        logs["ratedist"] = logs["rate"] + logs["distortion"]
        other = dict()

        return loss, logs, other

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        curr_step = self.idcs_to_compressors[optimizer_idx]

        # MODEL
        if curr_step == "main":
            loss, logs, other = self.step(batch)

            #! waiting for torch lightning #1243
            #! everyting in other should be detached
            self._save = other

        # ONLINE EVALUATOR
        elif curr_step == "online":
            loss, logs = self.online_evaluator(batch, self)

        # CODER
        elif curr_step == "coder":
            loss = self.rate_estimator.aux_loss()
            logs = dict(coder_loss=loss)

        else:
            raise ValueError(f"Unkown curr_step={curr_step}.")

        self.log_dict(
            {f"train/{self.stage}/{k}": v for k, v in logs.items()}, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):

        # TODO for some reason validation step for wandb logging after resetting is not correct
        loss, logs, _ = self.step(batch)

        if self.hparams.evaluation.featurizer.is_online:
            _, online_logs = self.online_evaluator(batch, self)
            logs.update(online_logs)

        self.log_dict(
            {f"val/{self.stage}/{k}": v for k, v in logs.items()},
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        # when commuicating only compute rate
        is_communicate = self.stage == "comm"
        loss, logs, _ = self.step(batch, is_rate_only=is_communicate)

        if self.hparams.evaluation.featurizer.is_online and not is_communicate:
            _, online_logs = self.online_evaluator(batch, self)
            logs.update(online_logs)

        self.log_dict(
            {f"test/{self.stage}/{k}": v for k, v in logs.items()},
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def on_test_epoch_start(self):
        # Make sure that you can actually use the coder during eval.
        self.rate_estimator.prepare_compressor_()

    def get_specific_parameters(self, mode):
        """Returns an iterator over the desired model parameters."""
        all_param = OrderedSet(self.parameters())
        coder_param = OrderedSet(self.rate_estimator.aux_parameters())
        rate_param = OrderedSet(self.rate_estimator.parameters()) - coder_param

        if self.hparams.evaluation.featurizer.is_online:
            online_param = OrderedSet(self.online_evaluator.aux_parameters())
        else:
            online_param = OrderedSet([])

        notrate_param = all_param - (coder_param | online_param | rate_param)
        if mode == "all":
            return all_param
        elif mode == "coder":
            return coder_param
        elif mode == "rate":
            return rate_param
        elif mode == "online":
            return online_param
        elif mode == "main":
            return notrate_param | rate_param
        elif mode == "notrate":
            return notrate_param
        else:
            raise ValueError(f"Unkown parameter mode={mode}.")

    def configure_optimizers(self):
        self.idcs_to_compressors = {}
        optimizers, schedulers = [], []
        n_opt = 0

        # COMPRESSOR OPTIMIZER
        self.idcs_to_compressors[n_opt] = "main"
        n_opt += 1
        append_optimizer_scheduler_(
            self.hparams.optimizer_feat,
            self.hparams.scheduler_feat,
            self.get_specific_parameters("main"),
            optimizers,
            schedulers,
            name="lr_featurizer",
        )

        # ONLINE EVALUATOR
        if self.hparams.evaluation.featurizer.is_online:
            self.idcs_to_compressors[n_opt] = "online"
            n_opt += 1
            append_optimizer_scheduler_(
                self.hparams.optimizer_online,
                self.hparams.scheduler_online,
                self.get_specific_parameters("online"),
                optimizers,
                schedulers,
                name="lr_online_eval",
            )

        # CODER OPTIMIZER
        coder_parameters = self.get_specific_parameters("coder")
        is_optimize_coder = len(coder_parameters) > 0

        if is_optimize_coder:
            self.idcs_to_compressors[n_opt] = "coder"
            n_opt += 1
            append_optimizer_scheduler_(
                self.hparams.optimizer_coder,
                self.hparams.scheduler_coder,
                coder_parameters,
                optimizers,
                schedulers,
                name="lr_coder",
            )

        return optimizers, schedulers

    def set_featurize_mode_(self):
        """Set as a featurizer."""

        # this ensures that nothing is persistent, i.e. will not be saved in checkpoint when
        # part of predictor
        for model in self.modules():
            params = dict(model.named_parameters(recurse=False))
            buffers = dict(model.named_buffers(recurse=False))
            for name, param in params.items():
                del model._parameters[name]
                model.register_buffer(name, param.data, persistent=False)

            for name, param in buffers.items():
                del model._buffers[name]
                model.register_buffer(name, param, persistent=False)

        self.freeze()
        self.eval()
        self.rate_estimator.make_pickable_()

    def on_load_checkpoint(self, checkpoint):
        # has to change the checkpoint to ensure that you can load a previous checkpoint even if
        # the number of groups in the optimizer changes. This is needed when using finetuning
        # as the number of parameters will change during training

        if self.trainer is not None:  # only if resuming
            #! waiting for https://github.com/PyTorchLightning/pytorch-lightning/issues/6891
            for callback in self.trainer.callbacks:
                if isinstance(callback, BaseFinetuning):
                    callback.loaded_epoch = checkpoint["epoch"]
                    callback.on_before_accelerator_backend_setup(self.trainer, self)
                    callback.on_train_epoch_start(self.trainer, self)
                    callback.loaded_epoch = -1
