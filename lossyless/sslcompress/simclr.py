import torch
from lossyless.compressors import CompressionModule
from lossyless.distortions import ContrastiveDistortion
from lossyless.rates import get_rate_estimator
from pl_bolts.models.self_supervised import SimCLR


class SimCLRCoder(SimCLR):
    """SimCLR with an entropy bottleneck for entropy coding. See `SimCLR` for documentation.

    Notes
    -----
    - The entropy coder should probably not be trained while SimCLR is trained, because the augmented
    images will not be sampled from real data generating porcess. Instead should first train (or finetune)
    SimCLR and then subsequently train the coder.

    Parameters
    ----------
    args, kwargs
        Additional arguments to SimCLR.

    rate_estimator : {"H_factorized","H_hyper"}, optional
        Rate estimator to use.
    """

    def __init__(self, *args, rate_estimator, weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_estimator = get_rate_estimator(
            rate_estimator, z_dim=self.hidden_mlp, **kwargs
        )
        self.distortion_estimator = ContrastiveDistortion(
            torch.nn.Identity,  # uses identity P_ZlX => take in tensors
            temperature=self.temperature,
            weight=weight,
        )

    def forward(self, x):

        h = self.encoder(x)[-1]

        # h_hat. shape: [n_z, batch_size, z_dim]
        h_hat, *_ = self.rate_estimator(h, None, None)

        return h_hat

    def shared_step(self, batch):
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        # final image in tuple is for online eval
        (img1, img2, _), y = batch

        # get h representations, bolts resnet returns a list
        h1 = self.encoder(img1)[-1]
        h2 = self.encoder(img2)[-1]

        #! changes are below this

        # our code assumes n_z samples
        h1 = h1.unsqueeze(0)
        h2 = h1.unsqueeze(0)

        h1_hat, rates1, r_logs1, _ = self.rate_estimator(h1, None, None)
        h2_hat, rates2, r_logs2, _ = self.rate_estimator(h2, None, None)
        rates = (rates1 + rates2) / 2

        # get z representations
        z1 = self.projection(h1_hat)
        z2 = self.projection(h2_hat)

        # x_pos, p_Zlx =  z2, z1 is ok because identity P_ZlX (use vectors directly)
        distortions, d_logs, _ = self.distortion_estimator(z1, z2, z1)

        loss, logs, _ = self.loss(rates, distortions)

        # to log (dict)
        logs.update(r_logs1)
        logs.update(r_logs2)
        logs.update(d_logs)
        logs.update(dict(zmin=z1.min(), zmax=z1.max(), zmean=z1.mean()))

        return loss, logs

    loss = CompressionModule.loss

    def training_step(self, batch, batch_idx):
        loss, logs = self.shared_step(batch)
        self.log_dict({f"train/{k}": v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.shared_step(batch)
        self.log_dict({f"valid/{k}": v for k, v in logs.items()})
        return loss
