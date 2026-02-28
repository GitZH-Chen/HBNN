"""
    Author: Ziheng Chen
    Implementation of Hyperbolic Busemann layers in
    @inproceedings{chen2026hbnn,
        title={Hyperbolic {Busemann} Neural Networks},
        author={Ziheng Chen and Bernhard Schölkopf and Nicu Sebe},
        booktitle={CVPR},
        year={2026},
    }
"""


import torch
import torch.nn as nn


class BLayer(nn.Module):
    def __repr__(self):
        attributes = []

        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    val_str = f"{value.item():.4f}"
                else:
                    val_str = f"shape={tuple(value.shape)}"
                attributes.append(f"{key}={val_str}")
            else:
                attributes.append(f"{key}={value}")

        for name, buffer in self.named_buffers(recurse=False):
            if buffer.numel() == 1:
                val_str = f"{buffer.item():.4f}"
            else:
                val_str = f"shape={tuple(buffer.shape)}"
            attributes.append(f"{name}={val_str}")

        for name, module in self.named_children():
            attributes.append(f"{name}={module.__repr__()}")

        return f"{self.__class__.__name__}({', '.join(attributes)})"
