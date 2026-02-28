# HBNN

This repository hosts the core implementation of **Hyperbolic Busemann Neural Networks (HBNN)**.

## Introduction

At this stage, `lib/bnn` mainly contains the key **Busemann layers** from the paper, and the training code will be released later.

If you find this project helpful, please consider citing:

```bibtex
@inproceedings{chen2026hbnn,
  title={Hyperbolic {Busemann} Neural Networks},
  author={Ziheng Chen and Bernhard Sch{\"o}lkopf and Nicu Sebe},
  booktitle={CVPR},
  year={2026}
}

@article{chen2025gyrobn_extension,
  title   = {Riemannian Batch Normalization: A Gyro Approach},
  author  = {Ziheng Chen and Xiaojun Wu and Bernhard Schölkopf and Nicu Sebe},
  journal = {arXiv preprint arXiv: 2509.07115},
  year    = {2025},
}
```

## What Is Included Now

The currently available core modules (under `lib/bnn`) are:

- `BMLR.py`: Busemann multinomial logistic regression (Lorentz / Poincare)
- `BFC.py`: Busemann fully-connected layer
- `Auxlayers.py`: wrapper layers (for synchronizing with learnable curvature / external manifold parameters)
- `Geometry/`: implementations of constant-curvature spaces

## Quick Usage

```python
from lib.bnn.BFC import BFC, Gyrobias
from lib.bnn.BMLR import BMLR

# Example: Lorentz Busemann FC
layer = BFC(
    in_dim=64,
    out_dim=128,
    metric="lorentz",
    K=-1.0,
    gyrobias=True,
)

# Example: Busemann classifier head
clf = BMLR(
    n_classes=10,
    dim=128,
    metric="lorentz",
    K=-1.0,
)
```


