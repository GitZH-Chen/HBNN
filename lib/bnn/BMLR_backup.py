import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt import ManifoldParameter

from lib.bnn.Geometry.base import EPS
from .Geometry import Sphere
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
        param_mode: str = "sep", # sep, joint
        metric: str = "lorentz",
        v_mode: str = "plain",  # riem, tri, plain
        g_mode: str = "exp_v", # none, exp, exp_v, softplus, linear
        precision: str = "float32", # float32, float64
        lambda_const: float = 1.0, # used if g_mode is linear
    ):
        super().__init__()
        if K >= 0.0:
            raise ValueError("Curvature K must be negative for hyperbolic geometry.")

        self.C = n_classes
        self.d = dim
        self.param_mode = param_mode.lower()
        self.metric = metric.lower()
        self.v_mode = v_mode.lower()
        self.use_bias = bias
        self.precision = precision.lower()
        if self.precision == "float64":
            self._precision_dtype = torch.float64
        elif self.precision == "float32":
            self._precision_dtype = torch.float32
        else:
            raise ValueError("precision must be either 'float32' or 'float64'.")
        self.g_mode = g_mode.lower()
        self.lambda_const = lambda_const
        if isinstance(K, torch.Tensor):
            self.K = K
        else:
            self.K = torch.tensor(K)

        if self.v_mode in {"tri"}:
            self._sphere = Sphere() # unit sphere for tri
        self._get_busemann()
        self._init_parameters()
        self._set_precision_dtype()

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
        self.weight_g = None

        if self.param_mode == "sep":
            if self.v_mode == "plain":
                self.weight_v = nn.Parameter(weight)
            elif self.v_mode == "riem":
                init_dir = weight / weight.norm(dim=-1, keepdim=True).clamp_min(EPS[weight.dtype])
                self.weight_v = ManifoldParameter(init_dir, manifold=geoopt.manifolds.Sphere())
            elif self.v_mode == "tri":
                self.weight_v = nn.Parameter(weight)
            else:
                raise RuntimeError("Invalid v_mode encountered during initialization.")
            # setting weight_g
            if self.g_mode == "none":
                self.weight_g = nn.Parameter(weight.norm(dim=-1))
            elif self.g_mode.startswith("exp"):
                if self.g_mode == "exp_v":
                    init = weight.norm(dim=-1).clamp_min(EPS[weight.dtype]).log()
                    self.weight_g = nn.Parameter(init)
                else:
                    self.weight_g = nn.Parameter(torch.zeros(self.C))
            elif self.g_mode == "softplus":
                # Softplus parameterization: g = softplus(theta)
                # Initialize so that softplus(theta0) = 1.0  -> theta0 = log(exp(1) - 1)
                # self.weight_g = nn.Parameter(weight.norm(dim=-1))
                _theta0 = torch.log(torch.expm1(torch.tensor(1.0)))
                self.weight_g = nn.Parameter(torch.full((self.C,), _theta0))
            elif self.g_mode == "linear":
                self.scaling_linear = nn.Linear(self.d, 1) 
            else:
                raise RuntimeError(f"Unsupported g_mode '{self.g_mode}'.")
        elif self.param_mode == "joint":
            self.weight = nn.Parameter(weight)
            
        else:
            raise ValueError("param_mode must be either 'sep' or 'joint'.")

        self.bias = nn.Parameter(torch.zeros(self.C), requires_grad=self.use_bias)

    def _set_precision_dtype(self) -> None:
        for param in self.parameters():
            param.data = param.data.to(self._precision_dtype)
        for name, buffer in self.named_buffers(recurse=False):
            setattr(self, name, buffer.to(self._precision_dtype))
        if isinstance(self.K, torch.Tensor) and self.K.dtype != self._precision_dtype:
            if isinstance(self.K, nn.Parameter):
                self.K.data = self.K.data.to(self._precision_dtype)
            else:
                self.K = self.K.to(self._precision_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self._precision_dtype:
            x = x.to(self._precision_dtype)

        if self.param_mode == "sep":
            v_unit_spatial = self._compute_unit_directions(self.weight_v)
            if self.g_mode == "none":
                scaling = self.weight_g
            elif self.g_mode.startswith("exp"):
                scaling = self.weight_g.exp()
            elif self.g_mode == "softplus":
                scaling = F.softplus(self.weight_g)
            elif self.g_mode == "linear":
                scaling = torch.sigmoid(self.scaling_linear(x)) * self.lambda_const
            else:
                raise RuntimeError(f"Unsupported g_mode '{self.g_mode}'.")
        else:
            weight_norm = self.weight.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
            v_unit_spatial = self.weight / weight_norm
            scaling = weight_norm.squeeze(-1)
        
        logits = self.busemann_logits(
            v_unit_spatial.transpose(0, 1),
            x,
            self.K,
            scaling,
            self.bias,
            EPS[x.dtype],
        )
        return logits
    
    def _compute_unit_directions(self, x: torch.Tensor) -> torch.Tensor:
        if self.v_mode == "plain":
            return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        elif self.v_mode == "riem":
            return x
        elif self.v_mode == "tri":
            #we can simply pass v to exp0, which only usees the space part.
            return self._sphere.exp0(x)
        else:
            raise RuntimeError(f"Unsupported v_mode '{self.v_mode}'.")

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
