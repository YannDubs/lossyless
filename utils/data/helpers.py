import logging
import math
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

import torch
from lossyless.helpers import to_numpy
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional as F_trnsf

logger = logging.getLogger(__name__)


def image_loader(path):
    """Load image and returns PIL."""
    if isinstance(path, Path):
        path = str(path.resolve())
    return default_loader(path)


class DownloadProgressBar(tqdm):
    """Progress bar for downlding files."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# Modified from https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
def download_url(url, save_dir):
    """Download a url to `save_dir`."""
    filename = url.split("/")[-1]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:

        urllib.request.urlretrieve(
            url, filename=save_dir / filename, reporthook=t.update_to
        )


def unzip(filename, is_rm=True):
    """Unzip file and optionally removes it."""
    filename = Path(filename)
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(filename.parent)
        if is_rm:
            filename.unlink()


def rotate(x, angle):
    """Rotate a 2D tensor by a certain angle (in degrees)."""
    angle = torch.as_tensor([angle * math.pi / 180])
    cos, sin = torch.cos(angle), torch.sin(angle)
    rot_mat = torch.as_tensor([[cos, sin], [-sin, cos]])
    return x @ rot_mat


def int_or_ratio(alpha, n):
    """Return an integer for alpha. If float, it's seen as ratio of `n`."""
    if isinstance(alpha, int):
        return alpha
    return int(alpha * n)


def npimg_resize(np_imgs, size):
    """Batchwise resizing numpy images."""
    if np_imgs.ndim == 3:
        np_imgs = np_imgs[:, :, :, None]

    torch_imgs = torch.from_numpy(np_imgs.transpose((0, 3, 1, 2))).contiguous()
    torch_imgs = F_trnsf.resize(torch_imgs, size=size)
    np_imgs = to_numpy(torch_imgs).transpose((0, 2, 3, 1))
    return np_imgs


# balancing weights can be computed by
# name="pets37"
# DataModule = get_datamodule(name)
# dm = DataModule(dataset_kwargs=dict(additional_target=None))
# dm.prepare_data()
# dm.setup()
# Y_test = [dm.test_dataset[i][1] for i in range(len(dm.test_dataset))]
# values, counts = np.unique(Y_test, return_counts=True)
# {str(v):sum(counts)/(len(counts) * c) for v,c in zip(values,counts)}
Pets37BalancingWeights = {
    "is_eval": True,  # says that computed on test set
    "0": 1.0118587975730833,
    "1": 0.9916216216216216,
    "2": 0.9916216216216216,
    "3": 1.1268427518427517,
    "4": 0.9916216216216216,
    "5": 1.022290331568682,
    "6": 0.9916216216216216,
    "7": 0.9916216216216216,
    "8": 0.9916216216216216,
    "9": 0.9916216216216216,
    "10": 0.9916216216216216,
    "11": 0.9916216216216216,
    "12": 0.9916216216216216,
    "13": 0.9916216216216216,
    "14": 0.9916216216216216,
    "15": 0.9916216216216216,
    "16": 1.0016380016380015,
    "17": 0.9916216216216216,
    "18": 0.9916216216216216,
    "19": 0.9916216216216216,
    "20": 0.9916216216216216,
    "21": 0.9916216216216216,
    "22": 0.9916216216216216,
    "23": 0.9916216216216216,
    "24": 1.0016380016380015,
    "25": 0.9916216216216216,
    "26": 0.9916216216216216,
    "27": 0.9916216216216216,
    "28": 0.9916216216216216,
    "29": 0.9916216216216216,
    "30": 0.9916216216216216,
    "31": 0.9916216216216216,
    "32": 1.0016380016380015,
    "33": 0.9916216216216216,
    "34": 1.1141815973276648,
    "35": 0.9916216216216216,
    "36": 0.9916216216216216,
}

Caltech101BalancingWeights = {
    "is_eval": True,  # says that computed on test set
    "0": 2.3858823529411763,
    "1": 0.0774637127578304,
    "2": 4.970588235294118,
    "3": 4.970588235294118,
    "4": 0.1364921254543007,
    "5": 3.508650519031142,
    "6": 2.485294117647059,
    "7": 3.7279411764705883,
    "8": 19.88235294117647,
    "9": 0.6086434573829532,
    "10": 0.8771626297577855,
    "11": 4.588235294117647,
    "12": 1.0844919786096257,
    "13": 0.9778206364513018,
    "14": 2.9823529411764707,
    "15": 4.588235294117647,
    "16": 0.6413662239089184,
    "17": 3.508650519031142,
    "18": 2.056795131845842,
    "19": 1.8639705882352942,
    "20": 0.774637127578304,
    "21": 3.508650519031142,
    "22": 1.5294117647058822,
    "23": 1.387140902872777,
    "24": 1.4911764705882353,
    "25": 2.9823529411764707,
    "26": 2.8403361344537816,
    "27": 2.2091503267973858,
    "28": 1.6120826709062004,
    "29": 2.7112299465240643,
    "30": 1.7042016806722688,
    "31": 1.5696594427244581,
    "32": 1.3254901960784313,
    "33": 1.754325259515571,
    "34": 2.5933503836317136,
    "35": 1.754325259515571,
    "36": 1.0844919786096257,
    "37": 0.14727668845315905,
    "38": 0.14727668845315905,
    "39": 1.6120826709062004,
    "40": 1.6120826709062004,
    "41": 3.976470588235294,
    "42": 14.911764705882353,
    "43": 14.911764705882353,
    "44": 2.8403361344537816,
    "45": 0.8644501278772379,
    "46": 0.8521008403361344,
    "47": 4.970588235294118,
    "48": 2.485294117647059,
    "49": 1.028397565922921,
    "50": 1.1929411764705882,
    "51": 59.64705882352941,
    "52": 1.754325259515571,
    "53": 1.065126050420168,
    "54": 0.7100840336134454,
    "55": 1.9240986717267552,
    "56": 1.1695501730103806,
    "57": 0.3508650519031142,
    "58": 1.2426470588235294,
    "59": 5.422459893048129,
    "60": 1.6568627450980393,
    "61": 4.588235294117647,
    "62": 5.964705882352941,
    "63": 1.0464396284829722,
    "64": 29.823529411764707,
    "65": 1.2966751918158568,
    "66": 0.07766544117647059,
    "67": 2.3858823529411763,
    "68": 11.929411764705883,
    "69": 6.627450980392157,
    "70": 3.508650519031142,
    "71": 7.455882352941177,
    "72": 3.976470588235294,
    "73": 2.5933503836317136,
    "74": 14.911764705882353,
    "75": 2.2091503267973858,
    "76": 1.1470588235294117,
    "77": 2.056795131845842,
    "78": 3.1393188854489162,
    "79": 5.964705882352941,
    "80": 1.8074866310160427,
    "81": 6.627450980392157,
    "82": 1.1045751633986929,
    "83": 2.2091503267973858,
    "84": 11.929411764705883,
    "85": 1.754325259515571,
    "86": 3.976470588235294,
    "87": 1.065126050420168,
    "88": 2.056795131845842,
    "89": 1.754325259515571,
    "90": 11.929411764705883,
    "91": 1.0844919786096257,
    "92": 3.1393188854489162,
    "93": 1.065126050420168,
    "94": 1.3254901960784313,
    "95": 0.2853926259499015,
    "96": 8.521008403361344,
    "97": 2.056795131845842,
    "98": 14.911764705882353,
    "99": 2.2941176470588234,
    "100": 6.627450980392157,
    "101": 1.988235294117647,
}
