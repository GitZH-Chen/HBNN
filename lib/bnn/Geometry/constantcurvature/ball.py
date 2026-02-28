import torch

from ..base import ConstantCurvature
from .utils import EPS, cosh, sinh, tanh, arcosh, arsinh, artanh, sinhdiv, divsinh

from .frechetmean.frechet import FrechetMeanBall

class Poincare(ConstantCurvature):
    """We use the following convention:  \| x \|^2 < - 1 / K"""
    def __init__(self, K=-1.0, edge_eps=1e-6):
        super().__init__(K=K)
        self.edge_eps = edge_eps

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5, dim=-1):
        """
        Check whether a point x lies inside the Poincaré ball: ‖x‖² < 1 / |K|
        """
        norm_sq = x.pow(2).sum(dim=dim, keepdim=True)
        radius_sq = 1.0 / (-self.K)
        ok = (norm_sq < radius_sq + atol).all()
        reason = None if ok else f"‖x‖² = {norm_sq.max().item():.5f} ≥ 1 / |K| = {radius_sq:.5f}"
        return ok, reason

    def _check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5, dim=-1):
        """
        For the Poincaré ball, every vector in R^n is a valid tangent vector.
        So we just check shape consistency.
        """
        ok = x.shape == u.shape
        reason = None if ok else f"x.shape = {x.shape} ≠ u.shape = {u.shape}"
        return ok, reason

    def random_normal(self, *size, mean=0, std=1,scale=1):
        """Sample from Gaussian in tangent space at origin and map to the manifold."""
        tangent = torch.randn(*size, device=self.K.device, dtype=self.K.dtype) * std + mean
        return self.exp0(tangent*scale,project=True)

    def random_tangent_origin(self, *size, mean=0, std=1, scale=1):
        tangent = torch.randn(*size, device=self.K.device, dtype=self.K.dtype) * std + mean
        return tangent*scale

    def sh_to_dim(self, sh):
        if hasattr(sh, '__iter__'):
            return sh[-1]
        else:
            return sh

    def dim_to_sh(self, dim):
        if hasattr(dim, '__iter__'):
            return dim[-1]
        else:
            return dim

    def zero(self, *shape):
        return torch.zeros(*shape)

    def zero_tan(self, *shape):
        return torch.zeros(*shape)

    def zero_like(self, x):
        return torch.zeros_like(x)

    def zero_tan_like(self, x):
        return torch.zeros_like(x)

    def lambda_x(self, x, keepdim=False):
        return 2 / (1 + self.K * x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(min=EPS[x.dtype])

    def inner(self, x, u, v, keepdim=False):
        return self.lambda_x(x, keepdim=True).pow(2) * (u * v).sum(dim=-1, keepdim=keepdim)

    def inner0(self, u, v, keepdim=False):
        return 4 * (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u):
        return u

    def projx(self, x):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS[x.dtype])
        maxnorm = (1 - self.edge_eps) / (-self.K).sqrt()
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def egrad2rgrad(self, x, u):
        return u / self.lambda_x(x, keepdim=True).pow(2)

    def exp(self, x, u, project=False):
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(min=EPS[x.dtype])
        second_term = (
            tanh((-self.K).sqrt() / 2 * self.lambda_x(x, keepdim=True) * u_norm) * u / ((-self.K).sqrt() * u_norm)
        )
        gamma_1 = self.gyroadd(x, second_term)
        if project:
            return self.projx(gamma_1)
        else:
            return gamma_1

    def exp0(self, u, project=False):
        sqrtK = (-self.K) ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), min=EPS[u.dtype])
        theta = sqrtK * u_norm
        gamma_1 = (tanh(theta) / theta) * u
        if project:
            return self.projx(gamma_1)
        else:
            return gamma_1

    def log(self, x, y):
        sub = self.gyroadd(-x, y)
        sub_norm = sub.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        lam = self.lambda_x(x, keepdim=True)
        res = 2 / ((-self.K).sqrt() * lam) * artanh((-self.K).sqrt() * sub_norm) * sub / sub_norm
        return res

    def log0(self, x):
        sqrtK = (-self.K) ** 0.5
        x_norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(EPS[x.dtype])
        y = sqrtK * x_norm
        scale = artanh(y) / y
        return scale * x

    def dist(self, x, y, squared=False, keepdim=False):
        dist = 2 * artanh((-self.K).sqrt() * self.gyroadd(-x, y).norm(dim=-1,keepdim=keepdim)) / (-self.K).sqrt()
        return dist.pow(2) if squared else dist

    def dist0(self, x, squared=False, keepdim=False):
        dist = 2 * artanh((-self.K).sqrt() * x.norm(dim=-1,keepdim=keepdim)) / (-self.K).sqrt()
        return dist.pow(2) if squared else dist

    def transp(self, x, y, u):
        return (
            self._gyration(y, -x, u)
            * self.lambda_x(x, keepdim=True)
            / self.lambda_x(y, keepdim=True)
        )

    def transpfrom0(self, x, u):
        return (2/self.lambda_x(x, keepdim=True)) * u

    def __str__(self):
        return 'Poincare Ball'

    def squeeze_tangent(self, x):
        return x

    def unsqueeze_tangent(self, x):
        return x

    def gyroadd(self, x, y):
        '''mobius addition on the Poincaré ball'''
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 - 2 * self.K * xy - self.K * y2) * x + (1 + self.K * x2) * y
        denom = 1 - 2 * self.K * xy + (self.K.pow(2)) * x2 * y2
        return num / denom.clamp_min(EPS[x.dtype])

    def gyroscalarprod(self, x, r):
        """Möbius scalar multiplication on the Poincaré ball."""
        K = self.K
        sqrt_minus_K = torch.sqrt(-K)
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        tanh_part = torch.tanh(r * torch.atanh(sqrt_minus_K * norm_x))
        scaled_x = (tanh_part / (sqrt_minus_K * norm_x)) * x
        return scaled_x

    def gyroinv(self, x):
        return -x

    def _gyration(self, u, v, w):
        u2 = u.pow(2).sum(dim=-1, keepdim=True)
        v2 = v.pow(2).sum(dim=-1, keepdim=True)
        uv = (u * v).sum(dim=-1, keepdim=True)
        uw = (u * w).sum(dim=-1, keepdim=True)
        vw = (v * w).sum(dim=-1, keepdim=True)
        a = - self.K.pow(2) * uw * v2 - self.K * vw + 2 * self.K.pow(2) * uv * vw
        b = - self.K.pow(2) * vw * u2 + self.K * uw
        d = 1 - 2 * self.K * uv + self.K.pow(2) * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(EPS[u.dtype])

    def gyrocoadd(self,x,y):
        return self.gyroadd(x,self._gyration(x,self.gyroinv(y),y))

    def frechet_mean(self,x,max_iter=1000,w=None):
        if w is None:
            w = torch.ones(x.shape[:-1]).to(x)
        return FrechetMeanBall.apply(x, w, self.K,max_iter)

