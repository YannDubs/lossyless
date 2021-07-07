# Lossy Compression for Lossless Prediction [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/lossyless/blob/main/LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

**Using:** [![Using](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YannDubs/lossyless/blob/main/notebooks/Hub.ipynb) 

**Training:** [![Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YannDubs/lossyless/blob/main/notebooks/minimal_code.ipynb)

This repostiory contains our implementation of the paper: [**Lossy Compression for Lossless Prediction**](https://arxiv.org/abs/2106.10800). That formalizes and empirically inverstigates unsupervised training for task-specific compressors.


## Using the compressor 

[![Using](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YannDubs/lossyless/blob/main/notebooks/Hub.ipynb)

If you want to use our compressor directly the easiest is to use the model from torch hub as seen in the google colab (or `notebooks/Hub.ipynb`) or th example below.

<details>
  <summary><b>Installation details</b></summary>

  ```bash
  pip install torch torchvision tqdm numpy compressai sklearn git+https://github.com/openai/CLIP.git
  ```

  Using pytorch`>1.7.1` : CLIP forces pytorch version `1.7.1`, this is because it needs this version to use JIT. If you don't need JIT (no JIT by default) you can alctually use more recent versions of torch and torchvision `pip install -U torch torchvision`. Make sure to update after having isntalled CLIP.

----------------------
</details>

```python
import time

import torch
from sklearn.svm import LinearSVC
from torchvision.datasets import STL10

DATA_DIR = "data/"

# list available compressors. b01 compresses the most (b01 > b005 > b001)
torch.hub.list('YannDubs/lossyless:main') 
# ['clip_compressor_b001', 'clip_compressor_b005', 'clip_compressor_b01']

# Load the desired compressor and transformation to apply to images (by default on GPU if available)
compressor, transform = torch.hub.load('YannDubs/lossyless:main','clip_compressor_b005')

# Load some data to compress and apply transformation
stl10_train = STL10(
    DATA_DIR, download=True, split="train", transform=transform
)
stl10_test = STL10(
    DATA_DIR, download=True, split="test", transform=transform
)

# Compresses the datasets and save them to file (this requires GPU)
# Rate: 1506.50 bits/img | Encoding: 347.82 img/sec
compressor.compress_dataset(
    stl10_train,
    f"{DATA_DIR}/stl10_train_Z.bin",
    label_file=f"{DATA_DIR}/stl10_train_Y.npy",
)
compressor.compress_dataset(
    stl10_test,
    f"{DATA_DIR}/stl10_test_Z.bin",
    label_file=f"{DATA_DIR}/stl10_test_Y.npy",
)

# Load and decompress the datasets from file the datasets (does not require GPU)
# Decoding: 1062.38 img/sec
Z_train, Y_train = compressor.decompress_dataset(
    f"{DATA_DIR}/stl10_train_Z.bin", label_file=f"{DATA_DIR}/stl10_train_Y.npy"
)
Z_test, Y_test = compressor.decompress_dataset(
    f"{DATA_DIR}/stl10_test_Z.bin", label_file=f"{DATA_DIR}/stl10_test_Y.npy"
)

# Downstream STL10 evaluation. Accuracy: 98.65% | Training time: 0.5 sec
clf = LinearSVC(C=7e-3)
start = time.time()
clf.fit(Z_train, Y_train)
delta_time = time.time() - start
acc = clf.score(Z_test, Y_test)
print(
    f"Downstream STL10 accuracy: {acc*100:.2f}%.  \t Training time: {delta_time:.1f} "
)
```


## Minimal training code

[![Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YannDubs/lossyless/blob/main/notebooks/minimal_code.ipynb)

If your goal is to look at a minimal version of the code to simply understand what is going on, I would highly recommend starting from `notebooks/minimal_compressor.ipynb` (or google colab link above). This is a notebook version of the code provided in Appendix E.7. of the paper, to quickly train and evaluate our compressor. 

<details>
  <summary><b>Installation details</b></summary>

  1. `pip install git+https://github.com/openai/CLIP.git`
  2. `pip uninstall -y torchtext` (probably not necessary but can cause issues if got installed as wrong pytorch version)
  3. `pip install scikit-learn==0.24.2 lightning-bolts==0.3.4 compressai==1.1.5 pytorch-lightning==1.3.8`

  Using pytorch`>1.7.1` : CLIP forces pytorch version `1.7.1` you should be able to use a more recent versions.  E.g.:
  1. `pip install git+https://github.com/openai/CLIP.git`
  2. `pip install -U torch torchvision scikit-learn lightning-bolts compressai pytorch-lightning`
</details>

## Results from the paper

We provide scripts to essentially replicate some results from the paper. The exact results will be a little different as we simplified and cleaned some of the code to help readability. All scripts can be found in `bin` and run using the command `bin/*/<experiment>.sh`.


<details>
<summary><b>Installation details</b></summary>

1. Clone repository
2. Install [PyTorch](https://pytorch.org/) >=  1.7
3. `pip install -r requirements.txt`

### Other installation
- For the bare minimum packages: use `pip install -r requirements_mini.txt` instead.
- For conda: use  `conda env update --file requirements/environment.yaml`.
- For docker: we provide a dockerfile at `requirements/Dockerfile`.

### Notes 

- CLIP forces pytorch version `1.7.1`, this is because it needs this version to use JIT. We don't use JIT so you can alctually use more recent versions of torch and torchvision `pip install -U torch torchvision`.
- For better logging: `hydra` and `pytorch lightning` logging don't work great together, to have a better logging experience you should comment out the folowing lines in `pytorch_lightning/__init__.py` :

```python
if not _root_logger.hasHandlers():
     _logger.addHandler(logging.StreamHandler())
     _logger.propagate = False
```

### Test installation

To test your installation and that everything works as desired you can run `bin/test.sh`, which will run an epoch of BICNE and VIC on MNIST.

----------------------

</details>

<details>
<summary><b>Scripts details</b></summary>

All scripts can be found in `bin` and run using the command `bin/*/<experiment>.sh`. This will save all results, checkpoints, logs... The most important results (including summary resutls and figures) will be saved at `results/exp_<experiment>`. Most important are the summarized metrics `results/exp_<experiment>*/summarized_metrics_merged.csv` and any figures `results/exp_<experiment>*/*.png`.

The key experiments that that do not require very large compute are:
- VIC/VAE on rotation invariant Banana distribution: `bin/banana/banana_viz_VIC.sh`
- VIC/VAE on augmentation invariant MNIST: `bin/mnist/augmist_viz_VIC.sh`
- CLIP experiments: `bin/clip/main_linear.sh`

By default all scripts will log results on [weights and biases](https://wandb.ai/site). If you have an account (or make one) you should set your username in `conf/user.yaml` after `wandb_entity:`, the passwod should be set directly in your environment variables. If you prefer not logging, you can use the command `bin/*/<experiment>.sh -a logger=csv` which changes (`-a` is for append) the default `wandb` logger to a `csv` logger.

Generally speaking you can change any of the parameters either directly in `conf/**/<file>.yaml` or by adding `-a` to the script. We are using [Hydra](https://hydra.cc/) to manage our configurations, refer to their documentation if something is unclear.

If you are using [Slurm](https://slurm.schedmd.com/documentation.html) you can submit directly the script on servers by adding a config file under `conf/slurm/<myserver>.yaml`, and then running the script as `bin/*/<experiment>.sh -s <myserver>`. For example configurations files for slurm see `conf/slurm/vector.yaml` or `conf/slurm/learnfair.yaml`. For more information check the documentation from [submitit's plugin](https://hydra.cc/docs/plugins/submitit_launcher) which we are using.


----------------------

</details>


### VIC/VAE on rotation invariant Banana

Command: 
```bash
bin/banana/banana_viz_VIC.sh
``` 

The following figures are saved automatically at `results/exp_banana_viz_VIC/**/quantization.png`. On the left we see the quantization of the Banana distribution by a standard compressor (called `VAE` in code but VC in paper). On the right, by our (rotation) invariant compressor (`VIC`).


<p float="left" align="middle">
  <img src="/results/exp_banana_viz_VIC/datafeat_banana_rot/feat_neural_feat/dist_VAE/enc_mlp_fancy/rate_H_factorized/optfeat_Adam_lr3.0e-04_w0.0e+00/schedfeat_expdecay1000/zdim_2/zs_1/beta_7.0e-02/seed_123/addfeat_None/quantization.png" width="47%" alt="Standard compression of Banana" />
  <img src="/results/exp_banana_viz_VIC/datafeat_banana_rot/feat_neural_feat/dist_VIC/enc_mlp_fancy/rate_H_factorized/optfeat_Adam_lr3.0e-04_w0.0e+00/schedfeat_expdecay1000/zdim_2/zs_1/beta_7.0e-02/seed_123/addfeat_None/quantization.png" width="47%"  alt="Invariant compression of Banana" /> 
</p>

### VIC/VAE on augmentend MNIST

Command: 
```bash
bin/banana/augmnist_viz_VIC.sh
``` 

The following figure is saved automatically at `results/exp_augmnist_viz_VIC/**/rec_imgs.png`. It shows source augmented MNIST images as well as the reconstructions using our invariant compressor.

![Invariant compression of augmented MNIST](/results/exp_augmnist_viz_VIC/datafeat_mnist_aug/feat_neural_rec/dist_VIC/enc_resnet18/rate_H_hyper/optfeat_AdamW_lr1.0e-03_w1.0e-05/schedfeat_expdecay100/zdim_128/zs_1/beta_1.0e-01/seed_123/addfeat_None/rec_imgs.png
)


### CLIP compressor


Command: 
```bash
bin/clip/main_small.sh
``` 

The following table comes directly from the results which are automatically saved at `results/exp_clip_bottleneck_linear_eval/**/datapred_*/**/results_predictor.csv`. It shows the result of compression from our CLIP compressor on many datasets.

|               | Cars196 | STL10 | Caltech101 | Food101 | PCam | Pets37 | CIFAR10 | CIFAR100 |
|---------------|:-------:|:-----:|:----------:|:-------:|:----:|:------:|:-------:|:--------:|
| Rate [bits]   |   1471  |  1342 |    1340    |   1266  | 1491 |  1209  |   1407  |   1413   |
| Test Acc. [%] |   80.3  |  98.5 |    93.3    |   83.8  | 81.1 |  88.8  |   94.6  |   79.0   |

Note: ImageNet is too large for training a SVM using SKlearn. You need to run MLP evaluation with `bin/clip/clip_bottleneck_mlp_eval`. Also you have to download ImageNet manually.


## Cite

You can read the full paper [here](https://arxiv.org/abs/2106.10800). Please cite our paper if you use our model:

```bibtex
@inproceedings{
    dubois2021lossy,
    title={Lossy Compression for Lossless Prediction},
    author={Yann Dubois and Benjamin Bloem-Reddy and Karen Ullrich and Chris J. Maddison},
    booktitle={Neural Compression: From Information Theory to Applications -- Workshop @ ICLR 2021},
    year={2021},
    url={https://arxiv.org/abs/2106.10800}
}
```