from pathlib import Path

import clip
import h5py
import numpy as np
import torch
import torch.multiprocessing
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from tqdm import tqdm

from lossyless.helpers import to_numpy
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
)

DATA_DIR = Path(__file__).parents[1].joinpath("data")


class ImagenetteDataset(ImageFolder):
    def __init__(self, root="data", train=True, transform=None, **kwargs):
        root = Path(root)
        suffix = "train" if train else "val"
        super().__init__(root / f"imagenette320/{suffix}", transform)


class ImagenetDataset(ImageFolder):
    def __init__(self, root="data", train=True, transform=None, **kwargs):
        root = Path(root)
        suffix = "train" if train else "val"
        super().__init__(root / f"imagenet256/{suffix}", transform)


class STL10Dataset(STL10):
    def __init__(self, *args, train=True, **kwargs):
        split = "train" if train else "test"
        super().__init__(*args, split=split, **kwargs)


def save_all_features(
    dataset,
    encoder,
    save_file,
    num_workers=8,
    batch_size=500,
    chunk_size=int(1e4),
    is_half=True,
    device="cpu",
):
    img = dataset[0][0].unsqueeze(0).to(device)
    if is_half:
        img = img.half()
    z_dim = encoder(img).shape[-1]
    chunk_size = min(len(dataset), chunk_size)

    with h5py.File(save_file, "a") as hf:
        Z = hf.create_dataset(
            "Z",
            shape=(len(dataset), z_dim),
            chunks=(chunk_size, z_dim),
            compression="gzip",
            dtype=np.float16,
        )
        Y = hf.create_dataset(
            "Y",
            shape=(len(dataset),),
            chunks=(chunk_size,),
            compression="gzip",
            dtype=int,
        )
        n_featurized = 0

        with torch.no_grad():
            for images, labels in tqdm(
                DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
            ):
                images = images.to(device)
                if is_half:
                    images = images.half()
                features = to_numpy(encoder(images))
                curr_n_feat = features.shape[0]
                Z[n_featurized : n_featurized + curr_n_feat, :] = features
                Y[n_featurized : n_featurized + curr_n_feat] = to_numpy(labels)
                n_featurized += curr_n_feat


def load_features(file):
    return h5py.File(file, "r")


def get_encoder_preprocess(model, data, is_half=True, device="cpu"):

    if model == "CLIP_ViT":
        entire_model, preprocess = clip.load("ViT-B/32", device)
        # if is_half:
        #     entire_model.half()
        encoder = entire_model.encode_image

    return encoder, preprocess


def get_featurized_data(
    Datasets, model, is_half=True, device="cpu", data_dir=DATA_DIR, **kwargs
):
    features = dict()
    data_dir = Path(data_dir)

    for data in Datasets.keys():

        folder_feat = data_dir / f"{model}/{data}"
        file_train = folder_feat / "train.h5"
        file_test = folder_feat / "test.h5"

        features[data] = dict()

        try:
            features[data]["train"] = load_features(file_train)
            features[data]["test"] = load_features(file_test)

        except (FileNotFoundError, IndexError, OSError):
            encoder, preprocess = get_encoder_preprocess(
                model, data, is_half=is_half, device=device
            )

            print(f"Featurizing {data}...")
            Dataset = Datasets[data]
            train = Dataset(data_dir, download=True, train=True, transform=preprocess)
            test = Dataset(data_dir, download=True, train=False, transform=preprocess)

            folder_feat.mkdir(parents=True, exist_ok=True)

            # Calculate the image features
            save_all_features(
                train, encoder, file_train, is_half=is_half, device=device, **kwargs
            )
            save_all_features(
                test, encoder, file_test, is_half=is_half, device=device, **kwargs
            )

            features[data]["train"] = load_features(file_train)
            features[data]["test"] = load_features(file_test)

        print(f"Done featurizing {data}")

    return features


if __name__ == "__main__":
    for model in ["CLIP_ViT"]:
        Datasets = dict(
            CIFAR10=CIFAR10,
            # CIFAR100=CIFAR100,
            # ImagenetteDataset=ImagenetteDataset,
            # ImagenetDataset=ImagenetDataset,
            # STL10=STL10Dataset
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        features = get_featurized_data(
            Datasets, model, device=device, is_half=True, num_workers=16
        )
        for data, hfs in features.items():
            hfs["train"].close()
            hfs["test"].close()
