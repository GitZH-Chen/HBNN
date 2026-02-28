import torch
import torch.nn as nn

from lib.bnn.Geometry.base import EPS
from . import BLayer

class BMLR(BLayer):
    """
    Busemann-based MLR on the Lorentz (hyperboloid) model.
    logit_k(x) = g_k * B^{v_k}(x) + b_k,
    where v_k is a unit tangent direction at the origin.
    """
    def __init__(
        self,
        n_classes: int,
        dim: int,
        K: float = -1.0,
        bias: bool = True,
        metric: str = "lorentz",
    ):
        super().__init__()
        if K >= 0.0:
            raise ValueError("Curvature K must be negative for hyperbolic geometry.")

        self.C = n_classes
        self.d = dim
        self.metric = metric.lower()
        self.use_bias = bias
        

        if isinstance(K, torch.Tensor):
            self.K = K
        else:
            self.K = torch.tensor(K)
        self._get_busemann()
        self._init_parameters()

    def _get_busemann(self):
        if self.metric == 'poincare':
            self.busemann_logits = _pbusemann_logits
        elif self.metric == 'lorentz':
            self.busemann_logits = _lbusemann_logits
        else:
            raise NotImplementedError(f"Unsupported metric '{self.metric}'. Expected 'poincare' or 'lorentz'.")

    def _init_parameters(self):
        weight = torch.empty(self.C, self.d)
        nn.init.normal_(weight, mean=0.0, std=self.d ** -0.5)
        self.weight_v = nn.Parameter(weight)
        self.weight_g = nn.Parameter(weight.norm(dim=-1).clamp_min(EPS[weight.dtype]).log())
        self.bias = nn.Parameter(torch.zeros(self.C, dtype=weight.dtype), requires_grad=self.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.weight_v.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        v_unit = self.weight_v / self.weight_v.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        scaling = self.weight_g.exp()
        
        logits = self.busemann_logits(
            v_unit.transpose(0, 1),
            x,
            self.K,
            scaling,
            self.bias,
            EPS[x.dtype],
        )
        return logits
    
#---- Busemann functions ----#
@torch.jit.script
def _lbusemann_logits(v_spatial: torch.Tensor,
                x: torch.Tensor,
                K: torch.Tensor,
                scaling: torch.Tensor,
                bias: torch.Tensor,
                eps: float) -> torch.Tensor:
    # Lorentz Busemann function
    # v_spatial is assumed to have shape (d, C)
    sqrt_c = (-K) ** 0.5
    spatial = x[..., 1:] # (..., d)
    time = x[..., :1]  # (..., 1)
    inner = spatial @ v_spatial - time  # (..., C)
    argument = (-sqrt_c) * inner
    busemann = torch.log(argument.clamp_min(eps)) / sqrt_c
    return -busemann * scaling + bias

@torch.jit.script
def _pbusemann_logits(v: torch.Tensor,
                x: torch.Tensor,
                K: torch.Tensor,
                scaling: torch.Tensor,
                bias: torch.Tensor,
                eps: float) -> torch.Tensor:
    # Poincare Busemann function (batch friendly)
    sqrt_c = (-K) ** 0.5                                #
    norm_v_sq = (v.pow(2)).sum(dim=0)                   # (C,)
    x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)      # (bs, 1)
    dot = x @ v                                         # (bs, C)

    numerator = norm_v_sq - 2 * sqrt_c * dot + (-K) * x_norm_sq
    denom = 1 + K * x_norm_sq
    ratio = numerator / denom.clamp_min(eps)
    busemann = torch.log(ratio) / sqrt_c
    return -busemann * scaling + bias
