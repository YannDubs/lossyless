import torch
import clip
from torchvision.datasets import CIFAR10, STL10

import tqdm
from torch.utils.data import DataLoader

import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from compressai.entropy_models import EntropyBottleneck
import math

from torchsummary import summary

import time
from pl_bolts.datamodules import SklearnDataModule

from sklearn.svm import LinearSVC


data_dir = "data/"
if torch.cuda.is_available():
    device, precision, gpus = "cuda", 16, 1
else:
    device, precision, gpus = "cpu", 32, 0

# pretrained CLIP
pretrained, preprocess = clip.load("ViT-B/32", device)

# train data
cifar = CIFAR10(data_dir, download=True, train=True, transform=preprocess)

label_map_CIFAR10 = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

label_map_STL10 = {
    0: 'airplane',
    1: 'bird',
    2: 'car',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'horse',
    7: 'monkey',
    8: 'ship',
    9: 'truck',
}

# eval data
stl10_train = STL10(data_dir, download=True, split="train", transform=preprocess)
stl10_test = STL10(data_dir, download=True, split="test", transform=preprocess)

def clip_featurize_data(dataset, device, pretrained):
    """Featurize a dataset using the pretrained CLIP model."""
    with torch.no_grad():
        Z, Y, W = [], [], []
        
        # make a codebook of label name.
        # if dataset_type == 'cifar10':
        #     label_map = label_map_CIFAR10
        # elif dataset_type == 'stl10':
        #     label_map = label_map_STL10
            
        # for i in range(len(label_map.keys())):
        #     W += [pretrained.encode_text(label_map[i].to(device).half()).cpu().numpy()]
            
        for x, y in tqdm.tqdm(DataLoader(dataset, batch_size=128, num_workers=16)):
            Z += [pretrained.encode_image(x.to(device).half()).cpu().numpy()]
            Y += [y.cpu().numpy()]
        
    return np.concatenate(Z), np.concatenate(Y)
    # return np.concatenate(Z), np.concatenate(Y), np.concatenate(W)

class ArrayCompressor(pl.LightningModule):
    """Compressor for any vectors, by using an entropy bottleneck and MSE distortion."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.bottleneck = EntropyBottleneck(self.hparams.z_dim)
        print(self.bottleneck)
        self.scaling = torch.nn.Parameter(torch.ones(self.hparams.z_dim))
        self.biasing = torch.nn.Parameter(torch.zeros(self.hparams.z_dim))
        self.is_updated = False

    def forward(self, batch):
        z, y = batch
        z = (z + self.biasing) * self.scaling.exp()
        z_hat, q_z = self.bottleneck(z.unsqueeze(-1).unsqueeze(-1))
        z_hat = z_hat.squeeze() / self.scaling.exp() - self.biasing
        return z_hat, q_z.squeeze(), y.squeeze()

    def _step(self, batch, *args, **kwargs):
        z_hat, q_z, _ = self(batch)
        rate = -torch.log(q_z).sum(-1).mean()  # Eq3 of Algorithm 1.
        distortion = torch.norm(batch[0] - z_hat, p=1, dim=-1).mean()  # 7 of Algorithm 1.
        self.log_dict(
            {"rate": rate / math.log(2), "distortion": distortion}, prog_bar=True
        )
        return distortion + self.hparams.lmbda * rate

    def similarity(self, batch):
        z, y = batch
        z_hat = z

    def step(self, batch, *args, **kwargs):
        z_hat, q_z, _ = self(batch)
        rate = -torch.log(q_z).sum(-1).mean()  # Eq3 of Algorithm 1.
        distortion = torch.norm(batch[0] - z_hat, p=1, dim=-1).mean()  # 7 of Algorithm 1.
        self.log_dict(
            {"rate": rate / math.log(2), "distortion": distortion}, prog_bar=True
        )
        return distortion + self.hparams.lmbda * rate

    def training_step(self, batch, _, optimizer_idx=0):
        return self.step(batch) if optimizer_idx == 0 else self.bottleneck.loss()

    def predict_step(self, batch, _, __):
        return self.compress(batch[0]), batch[1].cpu().numpy()

    def compress(self, z):
        if not self.is_updated:
            self.bottleneck.update(force=True)
            self.is_updated = True
        z = (z + self.biasing) * self.scaling.exp()
        return self.bottleneck.compress(z.unsqueeze(-1).unsqueeze(-1))

    def decompress(self, z_bytes):
        z_hat = self.bottleneck.decompress(z_bytes, [1, 1]).squeeze()
        return (z_hat / self.scaling.exp()) - self.biasing

    def configure_optimizers(self):
        param = [p for n, p in self.named_parameters() if not n.endswith(".quantiles")]
        quantile_param = [
            p for n, p in self.named_parameters() if n.endswith(".quantiles")
        ]
        optimizer = Adam(param, lr=self.hparams.lr)
        optimizer_coder = Adam(quantile_param, lr=self.hparams.lr)
        scheduler = StepLR(optimizer, self.hparams.lr_step)
        scheduler_coder = StepLR(optimizer_coder, self.hparams.lr_step)
        return [optimizer, optimizer_coder], [scheduler, scheduler_coder]
    
compressor = ArrayCompressor(z_dim=512, lmbda=4e-2, lr=1e-1, lr_step=2)

# start = time.time()
# Z_cifar, Y_cifar = clip_featurize_data(cifar, device, pretrained)
# data_kwargs = dict(
#     num_workers=16, batch_size=128, pin_memory=True, val_split=0.0, test_split=0
# )
# dm_cifar = SklearnDataModule(Z_cifar, Y_cifar, **data_kwargs)
# compressor = ArrayCompressor(z_dim=512, lmbda=4e-2, lr=1e-1, lr_step=2)
# trainer = pl.Trainer(gpus=gpus, precision=precision, max_epochs=10, logger=False)
# trainer.fit(compressor, datamodule=dm_cifar)
# print(f"Compressor trained in {(time.time() - start)/60:.0f} minutes.")

# def compress_data(trainer, dataset, device, pretrained, **kwargs):
#     """Compresses the data using an entropy coder."""
    
#     start = time.time()
#     Z, Y = clip_featurize_data(dataset, device, pretrained)
#     dm = SklearnDataModule(Z, Y, **kwargs)
#     out = trainer.predict(dataloaders=dm.train_dataloader())
    
#     Z_bytes = [o[0] for o in out]  # o[1] is label.
#     Y = np.concatenate([o[1] for o in out], axis=0)
    
#     flat_z = [i for batch in Z_bytes for i in batch]  # flat_z is z of each image.
#     coding_rate = sum([len(s) for s in flat_z]) * 8 / len(flat_z)  # 
#     sec_per_img = (time.time() - start) / len(flat_z)
#     return Z_bytes, Y, coding_rate, sec_per_img

# def decompress_data(compressor, Z_bytes):
#     """Compresses the data that was entropy coded."""
#     start = time.time()
#     with torch.no_grad():
#         Z_hat = [compressor.decompress(b).cpu().numpy() for b in Z_bytes]
#     sec_per_img = (time.time() - start) / len(Z_hat)
#     return np.concatenate(Z_hat), sec_per_img

# # entropy code evaluation data. Rate: 1703.6 bits, Compression: 230.2 img/sec
# Z_b_train, Y_train, *_ = compress_data(trainer, stl10_train, device, pretrained, **data_kwargs)
# Z_b_test, Y_test, rate, enc_time = compress_data(
#     trainer, stl10_test, device, pretrained, **data_kwargs
# )
# print(f"Bit-rate: {rate:.1f}. \t Compression: {1/enc_time:.1f} img/sec.")

# # Decompress data. Decoding: 2.8 img/sec (no batch processing)
# Z_train, _ = decompress_data(compressor, Z_b_train)
# Z_test, dec_time = decompress_data(compressor, Z_b_test)
# print(f"Decompression: {1/dec_time:.1f} img/sec.")

# # Downstream evaluation. Accuracy: 98.65%  	 Training time: 0.5
# clf = LinearSVC(C=4e-3)
# start = time.time()
# clf.fit(Z_train, Y_train)
# delta_time = time.time() - start
# acc = clf.score(Z_test, Y_test)
# print(
#     f"Downstream STL10 accuracy: {acc*100:.2f}%.  \t Training time: {delta_time:.1f} "
# )