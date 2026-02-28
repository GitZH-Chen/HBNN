import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import beta

from .Geometry import Sphere, Hyperboloid, Stereographic
from .Geometry.constantcurvature.utils.math_stereographic import _project as projx_stereographic
from .Geometry.constantcurvature.utils.math_hyperboloid import _calc_time
from .BMLR import _lbusemann_logits, _pbusemann_logits  
from .Geometry.base import EPS

from . import BLayer

class BFC(BLayer):
    def __init__(
        self,
        in_dim,
        out_dim,
        metric='poincare',
        bias=True,
        K=-1.,
        dropout=0,
        gyrobias=True,
        act = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_bias = bias
        self.metric = metric.lower()
        self.K = K
        self.dropout = dropout
        self.gyrobias = gyrobias

        if act is None:
            self.act = lambda x: x
        else:
            act_name = act.lower()
            if hasattr(F, act_name):
                self.act = getattr(F, act_name)
            else:
                raise ValueError(f"Unsupported activation '{act}'.")

        if isinstance(self.K, torch.Tensor):
            if torch.any(self.K >= 0.0):
                raise ValueError("Curvature K must be negative for hyperbolic geometry.")
        else:
            if self.K >= 0.0:
                raise ValueError("Curvature K must be negative for hyperbolic geometry.")
            self.K = torch.tensor(self.K)

        self._get_busemann_linear_and_manifold()
        self._init_parameters()

    def _get_busemann_linear_and_manifold(self):
        if self.metric == "poincare":
            self.busemann_linear = _poincare_busemann_linear
            self.manifold = Stereographic(K=self.K)
        elif self.metric == "lorentz":
            self.busemann_linear = _lorentz_busemann_linear
            self.manifold = Hyperboloid(K=self.K)   
        else:
            raise NotImplementedError(
                f"Unsupported metric '{self.metric}'. Expected 'poincare' or 'lorentz'."
            )

    def _init_parameters(self):
        gain = 1.
        weight = torch.empty(self.out_dim, self.in_dim).normal_( 
            mean=0, std=(2 * self.in_dim * self.out_dim) ** -0.5 * gain)
        self.weight_v = nn.Parameter(weight)
        self.weight_g = nn.Parameter(weight.norm(dim=-1).clamp_min(EPS[weight.dtype]).log())
        self.bias = nn.Parameter(torch.zeros(self.out_dim), requires_grad=self.is_bias)

        #---- gyrobias ---
        if self.gyrobias:
            self.tangent = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x):
        drop_weight = nn.functional.dropout(self.weight_v, self.dropout, training=self.training)
        v_unit = drop_weight / drop_weight.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        scaling = self.weight_g.exp()
        tmp = self.busemann_linear(
            x, 
            v_unit.transpose(0, 1), 
            scaling,
            self.bias, 
            self.K,
            EPS[x.dtype],
            self.act,
        )
        if self.gyrobias:
            if self.metric == "lorentz":
                tangent = torch.cat([self.tangent.new_zeros(1), self.tangent], dim=-1)
            else:
                tangent = self.tangent
            p = self.manifold.exp0(tangent)
            result = self.manifold.gyroadd(p, tmp)
        else:
            result = tmp
        return result

# @torch.jit.script
def _poincare_busemann_linear(x, v_unit, scaling, bias, K, eps: float, act):
    c=-K
    rc = c.sqrt()
    tmp = _pbusemann_logits(v_unit,x, K, scaling, bias, eps)
    v = act(tmp)
    omega = (rc * v).sinh() / rc
    return projx_stereographic(
        omega / (1 + (1 - K * omega.pow(2).sum(dim=-1, keepdim=True)).sqrt()),
        k=K,
        dim=-1,
    )

# @torch.jit.script
def _lorentz_busemann_linear(x, v_unit, scaling, bias, K, eps: float, act):
    sqrt_c = (-K).sqrt()
    tmp = _lbusemann_logits(v_unit, x, K, scaling, bias, eps)
    v = act(tmp)
    y_s = torch.sinh(sqrt_c * v) / sqrt_c
    res = torch.cat([_calc_time(y_s, K, eps), y_s], dim=-1)
    return res


#---- Gyrobias Layer ----
class Gyrobias(BLayer):
    def __init__(self, dim, metric="poincare", K=-1.0):
        super().__init__()
        self.dim = dim
        self.metric = metric.lower()
        self.K = K if isinstance(K, torch.Tensor) else torch.tensor(K)

        if self.metric == "poincare":
            self.manifold = Stereographic(K=self.K)
        elif self.metric == "lorentz":
            self.manifold = Hyperboloid(K=self.K)
        else:
            raise NotImplementedError(
                f"Unsupported metric '{self.metric}'. Expected 'poincare' or 'lorentz'."
            )

        self.tangent = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x):
        if self.metric == "lorentz":
            tangent = torch.cat([self.tangent.new_zeros(1), self.tangent], dim=0)
        else:
            tangent = self.tangent
        p = self.manifold.exp0(tangent)
        result = self.manifold.gyroadd(p, x)
        return result
