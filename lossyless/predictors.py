import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics.functional import accuracy

from .architectures import FlattenMLP, get_Architecture
from .distortions import mse_or_crossentropy_loss
from .helpers import Normalizer, Timer, append_optimizer_scheduler_, is_img_shape

__all__ = ["Predictor", "OnlineEvaluator"]


def get_featurizer_predictor(featurizer):
    """
    Helper function that returns a Predictor with correct featurizer. Cannot use partial because 
    need lighning module and cannot give as kwargs to load model because not pickable)
    """

    class FeatPred(Predictor):
        def __init__(self, hparams, featurizer=featurizer):
            super().__init__(hparams, featurizer)

    return FeatPred


class Predictor(pl.LightningModule):
    """Main network for downstream prediction."""

    def __init__(self, hparams, featurizer=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.is_clf = self.hparams.data.target_is_clf
        self.normalizer = torch.nn.Identity()

        if featurizer is not None:
            self.featurizer = featurizer
            self.featurizer.freeze()
            self.featurizer.eval()
            pred_in_shape = featurizer.out_shape

            is_normalize = self.hparams.data.kwargs.dataset_kwargs.is_normalize
            if is_img_shape(pred_in_shape) and is_normalize:
                # reapply normalization because lost during compression
                self.normalizer = Normalizer(self.hparams.data.dataset)

        else:
            self.featurizer = torch.nn.Identity()
            pred_in_shape = self.hparams.data.shape

        cfg_pred = self.hparams.predictor
        Architecture = get_Architecture(cfg_pred.arch, **cfg_pred.arch_kwargs)
        self.predictor = Architecture(pred_in_shape, self.hparams.data.target_shape)

    @auto_move_data  # move data on correct device for inference
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

    def step(self, batch):
        x, y = batch

        # shape: [batch_size,  *target_shape]
        Y_hat, logs = self(x, is_return_logs=True)

        # Shape: [batch, 1]
        loss = self.loss(Y_hat, y)

        # Shape: []
        loss = loss.mean()

        logs["loss"] = loss
        if self.is_clf:
            logs["acc"] = accuracy(Y_hat.argmax(dim=-1), y)

        return loss, logs

    def loss(self, Y_hat, y):
        """Compute the MSE or cross entropy loss."""
        return mse_or_crossentropy_loss(Y_hat, y, self.is_clf)

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict({f"train/pred/{k}": v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict(
            {f"val/pred/{k}": v for k, v in logs.items()}, on_epoch=True, on_step=False
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict(
            {f"test/pred/{k}": v for k, v in logs.items()}, on_epoch=True, on_step=False
        )
        return loss

    def configure_optimizers(self):

        optimizers, schedulers = [], []

        append_optimizer_scheduler_(
            self.hparams.optimizer_pred,
            self.hparams.scheduler_pred,
            self.parameters(),
            optimizers,
            schedulers,
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

    is_classification : bool, optional
        Whether or not the task is a classification one.

    kwargs:
        Additional kwargs to the MLP predictor.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        is_classification=True,
        n_hid_layers=1,
        hid_dim=1024,
        norm_layer="batchnorm",
        dropout_p=0.2,
    ):
        super().__init__()
        self.model = FlattenMLP(
            in_dim,
            out_dim,
            n_hid_layers=n_hid_layers,
            hid_dim=hid_dim,
            norm_layer=norm_layer,
            dropout_p=dropout_p,
        )
        self.is_classification = is_classification

    def parameters(self):
        """No parameters should be trained by main optimizer."""
        return iter(())

    def aux_parameters(self):
        """Return iterator over parameters."""
        for m in self.children():
            for p in m.parameters():
                yield p

    def forward(self, batch, encoder):
        x, (y, _) = batch  # only return the real label

        with torch.no_grad():
            # Shape: [batch, z_dim]
            z = encoder(x, is_features=True)

        z = z.detach()

        with Timer() as inference_timer:
            # Shape: [batch, *Y_shape]
            Y_hat = self.model(z)

        # Shape: [batch, 1]
        loss = mse_or_crossentropy_loss(Y_hat, y, self.is_classification)

        # Shape: []
        loss = loss.mean()

        logs = dict(online_loss=loss, inference_time=inference_timer.duration)
        if self.is_classification:
            logs["online_acc"] = accuracy(Y_hat.argmax(dim=-1), y)

        return loss, logs
