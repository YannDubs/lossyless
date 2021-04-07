import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from torchmetrics.functional import accuracy

from .architectures import get_Architecture
from .distortions import prediction_loss
from .helpers import (
    Normalizer,
    Timer,
    append_optimizer_scheduler_,
    is_img_shape,
    to_numpy,
)

__all__ = ["Predictor", "OnlineEvaluator"]

logger = logging.getLogger(__name__)


def get_featurizer_predictor(featurizer):
    """
    Helper function that returns a Predictor with correct featurizer. Cannot use partial because 
    need lighning module and cannot give as kwargs to load model because not pickable)
    """

    class FeatPred(Predictor):
        def __init__(self, hparams, featurizer=featurizer):
            super().__init__(hparams, featurizer=featurizer)

    return FeatPred


class Predictor(pl.LightningModule):
    """Main network for downstream prediction."""

    def __init__(self, hparams, featurizer=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.is_clf = self.hparams.data.target_is_clf

        if featurizer is not None:
            # ensure not saved in checkpoint and frozen
            featurizer.set_featurize_mode_()
            self.featurizer = featurizer
            pred_in_shape = featurizer.out_shape

            is_normalize = self.hparams.data.kwargs.dataset_kwargs.is_normalize
            if is_img_shape(pred_in_shape) and is_normalize:
                # reapply normalization because lost during compression
                self.normalizer = Normalizer(self.hparams.data.dataset)
            else:
                self.normalizer = torch.nn.Identity()

        else:
            self.featurizer = torch.nn.Identity()
            self.normalizer = torch.nn.Identity()  # already normalized if needed
            pred_in_shape = self.hparams.data.shape

        cfg_pred = self.hparams.predictor
        Architecture = get_Architecture(cfg_pred.arch, **cfg_pred.arch_kwargs)
        self.predictor = Architecture(pred_in_shape, self.hparams.data.target_shape)

    # @auto_move_data  # move data on correct device for inference
    def forward(self, x, is_logits=True, is_return_logs=False):
        """Perform prediction for `x`.

        Parameters
        ----------
        x : torch.Tensor of shape=[batch_size, *data.shape]
            Data to represent.

        is_logit : bool, optional
            Whether to return the logits instead of classes probablity in case you are using using
            classification.

        is_return_logs : bool, optional 
            Whether to return a dictionnary to log in addition to Y-pred.

        Returns
        -------
        Y_pred : torch.Tensor of shape=[batch_size, *target_shape]

        if is_return_logs:
            logs : dict
                Dictionary of values to log.
        """
        with torch.no_grad():
            # shape: [batch_size,  *featurizer.out_shape]
            features = self.featurizer(x)  # can be z_hat or x_hat
            features = self.normalizer(features)

        features = features.detach()  # shouldn't be needed

        with Timer() as inference_timer:
            # shape: [batch_size,  *target_shape]
            Y_pred = self.predictor(features)

        if not is_logits and self.is_clf:
            out = Y_pred.softmax(-1)
        else:
            out = Y_pred

        if is_return_logs:
            batch_size = Y_pred.size(0)
            logs = dict(inference_time=inference_timer.duration / batch_size)
            return out, logs

        return out

    def add_balanced_logs(self, loss, y, Y_hat, logs):
        mapper = self.hparams.data.balancing_weights
        assert mapper["is_eval"]  # currently only during val

        sample_weights = torch.tensor([mapper[yi.item()] for yi in y], device=y.device)
        logs["balanced_loss"] = (loss * sample_weights).mean()

        if self.is_clf:
            is_same = (Y_hat.argmax(dim=-1) == y).float()
            balanced_acc = (is_same * sample_weights).mean()
            logs["balanced_acc"] = balanced_acc
            logs["balanced_err"] = 1 - logs["balanced_acc"]

        return logs

    def step(self, batch):
        x, y = batch

        # shape: [batch_size,  *target_shape]
        Y_hat, logs = self(x, is_return_logs=True)

        # Shape: [batch, 1]
        loss, loss_logs = self.loss(Y_hat, y)

        if not self.training and len(self.hparams.data.balancing_weights) > 0:
            loss_logs = self.add_balanced_logs(loss, y, Y_hat, loss_logs)

        # Shape: []
        loss = loss.mean()

        logs.update(loss_logs)
        logs["loss"] = loss
        if self.is_clf:
            logs["acc"] = accuracy()
            logs["err"] = 1 - logs["acc"]

        return loss, logs

    def loss(self, Y_hat, y):
        """Compute the MSE or cross entropy loss."""

        loss = prediction_loss(
            Y_hat,
            y,
            self.is_clf,
            agg_over_tasks="mean",
            **self.hparams.predictor.loss_kwargs,
        )

        logs = {}
        # assumes that shape is (Y_dim, n_tasks) or (Y_dim) for single task
        # if single task std will be nan which is ok
        for agg_over_tasks in ["max", "std"]:
            agg = prediction_loss(
                Y_hat.detach(),
                y,
                self.is_clf,
                agg_over_tasks=agg_over_tasks,
                **self.hparams.predictor.loss_kwargs,
            )
            logs[f"tasks_{agg_over_tasks}"] = agg.mean()

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict({f"train/pred/{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict(
            {f"val/pred/{k}": v for k, v in logs.items()},
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict(
            {f"test/pred/{k}": v for k, v in logs.items()},
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def on_test_epoch_start(self):
        self.featurizer.on_test_epoch_start()

    def configure_optimizers(self):

        optimizers, schedulers = [], []

        append_optimizer_scheduler_(
            self.hparams.optimizer_pred,
            self.hparams.scheduler_pred,
            self.parameters(),
            optimizers,
            schedulers,
            name="lr_predictor",
        )

        return optimizers, schedulers


class OnlineEvaluator(torch.nn.Module):
    """
    Attaches MLP/linear predictor for evaluating the quality of a representation as usual in self-supervised.

    Notes
    -----
    -  generalizes `pl_bolts.callbacks.ssl_online.SSLOnlineEvaluator` for multilabel clf and regression
    and does not use a callback as pytorch lightning doesn't work well with trainable callbacks.

    Parameters
    ----------
    in_dim : int
        Input dimension.

    y_shape : tuple of in
        Shape of the output

    Architecture : nn.Module
        Module to be instantiated by `Architecture(in_shape, out_dim)`.

    kwargs:
        Additional kwargs to `prediction_loss`.
    """

    def __init__(self, in_dim, out_dim, Architecture, **kwargs):
        super().__init__()
        self.model = Architecture(in_dim, out_dim)
        self.kwargs = kwargs

    def aux_parameters(self):
        """Return iterator over parameters."""
        for m in self.children():
            for p in m.parameters():
                yield p

    def forward(self, batch, encoder):
        x, y = batch

        if isinstance(y, (tuple, list)):
            y = y[0]  # only return the real label assumed to be first

        with torch.no_grad():
            # Shape: [batch, z_dim]
            z = encoder(x, is_features=True)

        z = z.detach()

        with Timer() as inference_timer:
            # Shape: [batch, *Y_shape]
            Y_hat = self.model(z)

        # Shape: [batch, 1]
        loss = prediction_loss(Y_hat, y, **self.kwargs)

        # Shape: []
        loss = loss.mean()

        logs = dict(online_loss=loss, inference_time=inference_timer.duration)
        if self.is_classification:
            logs["online_acc"] = accuracy(Y_hat.argmax(dim=-1), y)
            logs["online_err"] = 1 - logs["online_acc"]

        return loss, logs
