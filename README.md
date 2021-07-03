# Lossy Compression for Lossless Prediction [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/lossyless/blob/main/LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

[![Minimal script](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

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

## Run

### Using the compressor 

... TODO pytorch hub... 


### Minimal code

If your goal is to look at a minimal version of the code to simply understand what is going on, I would highly recommend starting by `minimal_compressor.py`. This is a script version of the code provided in Appendix E.7. of the paper, to quickly train and evaluate our compressor. To run it simply use 

```bash
python minimal_compressor.py
```

### Replicating results

### Modifying configs

## Output

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
