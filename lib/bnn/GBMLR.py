import torch
import torch.nn as nn

from lib.geoopt.manifolds.stereographic.math import arsinh, artanh
from lib.bnn.Geometry.constantcurvature import Stereographic, Hyperboloid
from lib.bnn.Geometry.base import EPS

class GyroBusemannMLR(nn.Module):
    """Hyperbolic multinomial logistic regression layer: g_k B(v_k, \ominus p_k \oplus x)."""
    def __init__(
        self,
        n_classes: int,
        dim: int,
        K: float = -1.0,
        metric: str = "poincare",
        param_mode: str = "sep", # sep,joint
    ):
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.K = K
        self.metric = metric.lower()
        self.param_mode = param_mode.lower()
        assert self.K < 0.0, "Curvature K must be negative for hyperbolic geometry."      
        if self.param_mode not in {"sep", "joint"}:
            raise ValueError("param_mode must be either 'sep' or 'joint'.")
        self.getmanifolds()

        INIT_EPS = 1e-3
        ambient_dim = self.manifold.dim_to_sh(self.dim)
        self.point = nn.Parameter(torch.randn(self.n_classes, ambient_dim) * INIT_EPS) #[c,d] for poincare, [c,d+1] for lorentz

        init_weight = torch.randn(self.n_classes, self.dim) * INIT_EPS
        if self.param_mode == "sep":
            init_norm = init_weight.norm(dim=-1)
            self.weight = nn.Parameter(init_weight) #[c,d]
            self.weight_g = nn.Parameter(init_norm) #[c]
        elif self.param_mode == "joint":
            self.weight = nn.Parameter(init_weight) #[c,d]
        else:
            raise NotImplementedError
        
    def getmanifolds(self):
        if self.metric == 'poincare':
            self.manifold = Stereographic(K=self.K)
        elif self.metric == 'lorentz':
            self.manifold = Hyperboloid(K=self.K)
        else:
            raise NotImplementedError(f"Unsupported metric '{self.metric}'. Expected 'poincare' or 'lorentz'.")
    
    def forward(self, x):
        p = self.manifold.exp0(self.point)
        item = self.manifold.gyroadd(self.manifold.gyroinv(p), x.unsqueeze(1)) # [b,c,d]

        weight_norm = self.weight.norm(dim=-1, keepdim=True).clamp_min(EPS[self.weight.dtype])
        unit_direction = self.weight / weight_norm
        if self.param_mode == "sep":
            scaling = self.weight_g
        else:
            scaling = weight_norm.squeeze(-1)

        busemann_values = self.manifold.busemann(unit_direction, item)
        logits = -busemann_values * scaling
        return logits

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

        return f"{self.__class__.__name__}({', '.join(attributes)}) \n"
