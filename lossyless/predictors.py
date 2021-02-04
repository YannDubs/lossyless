import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy

from .architectures import FlattenMLP
from .distortions import mse_or_crossentropy_loss

__all__ = ["Predictor", "OnlineEvaluator"]


class Predictor(pl.LightningModule):
    """Main network for downstream prediction."""

    # TODO

    # def __init__(self, hparams, featurizer=torch.nn.Identity()):
    #     super().__init__()
    #     self.save_hyperparameters(hparams)

    #     self.featurizer = featurizer

    #     cfg_pred = self.hparams.predictor
    #     Architecture = get_Architecture(cfg_pred.arch, **cfg_pred.arch_kwargs)
    #     self.mapper = Architecture(in_shape, self.hparams.data.target_shape)
    #     self.predictor =

    #     def get_encoder(self):
    #     """Return encoder: a mapping to a torch.Distribution (conditional distribution)."""
    #     cfg_enc = self.hparams.encoder
    #     return CondDist(
    #         self.hparams.data.shape,
    #         cfg_enc.z_dim,
    #         family=cfg_enc.fam,
    #         Architecture=get_Architecture(cfg_enc.arch, **cfg_enc.arch_kwargs),
    #         **cfg_enc.fam_kwargs,
    #     )


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
            z = encoder(x)

        z = z.detach()

        # Shape: [batch, *Y_shape]
        Y_hat = self.model(z)

        # Shape: [batch, 1]
        loss = mse_or_crossentropy_loss(Y_hat, y, self.is_classification)

        # Shape: []
        loss = loss.mean()

        logs = dict(online_loss=loss)
        if self.is_classification:
            logs["online_acc"] = accuracy(Y_hat.argmax(dim=-1), y)

        return loss, logs
