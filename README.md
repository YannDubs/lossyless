# Lossy Compression for Lossless Prediction [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/lossyless/blob/main/LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

[![Minimal code](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YannDubs/lossyless/blob/main/minimal_code.ipynb)

This repostiory contains our implementation of the paper: [**Lossy Compression for Lossless Prediction**](https://arxiv.org/abs/2106.10800). That formalizes and empirically inverstigates unsupervised training for task-specific compressors.

## Install

0. Clone repository
1. Install [PyTorch](https://pytorch.org/) >=  1.7
2. `pip install -r requirements.txt`

Nota Bene: 
- For conda: use  `conda env update --file requirements/environment.yaml`.
- For the bare minimum packages in pip use `pip install -r requirements_mini.txt`
- For docker: we provide a dockerfile at `requirements/Dockerfile`.
- For better logging: `hydra` and `pytorch lightning` logging don't work great together, to have a better logging experience you should comment out the folowing lines in `pytorch_lightning/__init__.py` :

```python
if not _root_logger.hasHandlers():
     _logger.addHandler(logging.StreamHandler())
     _logger.propagate = False
```

## Test

To test your installation and that everything works as desired you can run `bin/test.sh`, which will run an epoch of BICNE and VIC on MNIST.

## Run

### Using the compressor 

... TODO pytorch hub... 


### Minimal code

If your goal is to look at a minimal version of the code to simply understand what is going on, I would highly recommend starting by `minimal_compressor.py`. This is a script version of the code provided in Appendix E.7. of the paper, to quickly train and evaluate our compressor. To run it simply use 

```bash
python minimal_compressor.py
```

### Results from the paper

We provide scripts to essentially replicate many of the results from the paper. The exact results will be a little different as we simplified and cleaned some of the code to help readability.

All scripts can be found in `bin` and run using the command `bin/*/<experiment>.sh`. This will save all results, checkpoints, logs... The most important results (including summary resutls and figures) will be saved at `results/exp_<experiment>`. Most important are the summarized metrics `results/exp_<experiment>*/summarized_metrics_merged.csv` and any figures `results/exp_<experiment>*/*.png`.

The key experiments that that do not require very large compute are:
- VIC/VAE on rotation invariant Banana distribution: `bin/banana/banana_viz_VIC.sh`
- VIC/VAE on augmentation invariant MNIST: `bin/mnist/augmist_viz_VIC.sh`
- CLIP experiments: `bin/clip/main_linear.sh`

By default all scripts will log results on [weights and biases](https://wandb.ai/site). If you have an account (or make one) you should set your username in `conf/user.yaml` after `wandb_entity:`, the passwod should be set directly in your environment variables. If you prefer not logging, you can use the command `bin/*/<experiment>.sh -a logger=csv` which changes (`-a` is for append) the default `wandb` logger to a `csv` logger.

Generally speaking you can change any of the parameters either directly in `conf/**/<file>.yaml` or by adding `-a` to the script. We are using [Hydra](https://hydra.cc/) to manage our configurations, refer to their documentation if something is unclear.

If you are using [Slurm](https://slurm.schedmd.com/documentation.html) you can submit directly the script on servers by adding a config file under `conf/slurm/<myserver>.yaml`, and then running the script as `bin/*/<experiment>.sh -s <myserver>`. For example configurations files for slurm see `conf/slurm/vector.yaml` or `conf/slurm/learnfair.yaml`. For more information check the documentation from [submitit's plugin](https://hydra.cc/docs/plugins/submitit_launcher) which we are using.





## Cite
```
@inproceedings{
dubois2021lossy,
title={Lossy Compression for Lossless Prediction},
author={Yann Dubois and Benjamin Bloem-Reddy and Karen Ullrich and Chris J. Maddison},
booktitle={Neural Compression: From Information Theory to Applications -- Workshop @ ICLR 2021},
year={2021},
url={https://arxiv.org/abs/2106.10800}
}
```
