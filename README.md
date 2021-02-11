# Lossy Compression for Lossless Prediction [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/Neural-Process-Family/blob/master/LICENSE) [![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

WIP

## Install

0. Clone repository
1. Install [PyTorch](https://pytorch.org/) >=  1.7
2. `pip install -r requirements.txt`

Nota Bene: 
- if you prefer I also provide a `Dockerfile` to install the necessary packages.
- `hydra` and `pytorch lightning` logging don't work great together (specifically pytorch lighning logs go to stderr and don't propagate). To have a better logging experience you should comment (or delete) out the folowing lines in `pytorch_lightning/__init__.py` :

```python
if not _root_logger.hasHandlers():
     _logger.addHandler(logging.StreamHandler())
     _logger.propagate = False
```

