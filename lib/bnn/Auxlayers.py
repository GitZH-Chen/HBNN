import torch
import torch.nn as nn

from .BFC import BFC, Gyrobias
from .BMLR import BMLR


class LorentzBFCWrapper(nn.Module):
    """
    Dynamically injects the current Lorentz curvature into a BFC layer so
    gradients can flow to the manifold parameter `k`.
    """

    def __init__(
        self,
        manifold,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
        gyrobias: bool = True,
        act=None,
    ):
        super().__init__()
        self.manifold = manifold

        init_K = (-1.0 / manifold.k.detach()).clone().to(
            dtype=manifold.k.dtype, device=manifold.k.device
        )
        self.layer = BFC(
            in_dim=in_dim,
            out_dim=out_dim,
            metric="lorentz",
            bias=bias,
            K=init_K,
            dropout=dropout,
            gyrobias=gyrobias,
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = -1.0 / self.manifold.k
        self.layer.K = K
        if hasattr(self.layer, "manifold") and hasattr(self.layer.manifold, "K"):
            self.layer.manifold.K = K
        return self.layer(x)


class LorentzBMLRWrapper(nn.Module):
    """
    Wraps BMLR to recompute the Lorentz curvature every forward pass,
    ensuring autograd can reach the shared manifold parameter.
    """

    def __init__(
        self,
        manifold,
        n_classes: int,
        dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.manifold = manifold

        init_K = (-1.0 / manifold.k.detach()).clone().to(
            dtype=manifold.k.dtype, device=manifold.k.device
        )
        self.layer = BMLR(
            n_classes=n_classes,
            dim=dim,
            K=init_K,
            bias=bias,
            metric="lorentz",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = -1.0 / self.manifold.k
        self.layer.K = K
        return self.layer(x)


class LorentzGyrobiasWrapper(nn.Module):
    """
    Wraps Gyrobias so the curvature parameter is refreshed in each forward pass.
    """

    def __init__(self, manifold, dim: int):
        super().__init__()
        self.manifold = manifold

        init_K = (-1.0 / manifold.k.detach()).clone().to(
            dtype=manifold.k.dtype, device=manifold.k.device
        )
        self.layer = Gyrobias(dim=dim, metric="lorentz", K=init_K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = -1.0 / self.manifold.k
        self.layer.K = K
        if hasattr(self.layer, "manifold") and hasattr(self.layer.manifold, "K"):
            self.layer.manifold.K = K
        return self.layer(x)

#--- Poincare Wrappers ---#

class PoincareBFCWrapper(nn.Module):
    """
    Keeps the curvature of an associated Poincaré manifold in sync with a BFC layer.
    """

    def __init__(
        self,
        manifold,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
        gyrobias: bool = True,
        act=None,
    ):
        super().__init__()
        self.manifold = manifold

        init_K = manifold.k.detach().clone()
        self.layer = BFC(
            in_dim=in_dim,
            out_dim=out_dim,
            metric="poincare",
            bias=bias,
            K=init_K,
            dropout=dropout,
            gyrobias=gyrobias,
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.manifold.k
        self.layer.K = K
        if hasattr(self.layer, "manifold") and hasattr(self.layer.manifold, "K"):
            self.layer.manifold.K = K
        return self.layer(x)


class PoincareBMLRWrapper(nn.Module):
    """
    Wraps BMLR for the Poincaré ball so curvature tracks a shared geoopt manifold.
    """

    def __init__(
        self,
        manifold,
        n_classes: int,
        dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.manifold = manifold

        init_K = manifold.k.detach().clone().to(
            dtype=manifold.k.dtype, device=manifold.k.device
        )
        self.layer = BMLR(
            n_classes=n_classes,
            dim=dim,
            K=init_K,
            bias=bias,
            metric="poincare",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.layer.K = self.manifold.k
        return self.layer(x)


class PoincareGyrobiasWrapper(nn.Module):
    """
    Synchronises the curvature of a Gyrobias layer with a Poincaré manifold instance.
    """

    def __init__(self, manifold, dim: int):
        super().__init__()
        self.manifold = manifold

        init_K = manifold.k.detach().clone().to(
            dtype=manifold.k.dtype, device=manifold.k.device
        )
        self.layer = Gyrobias(dim=dim, metric="poincare", K=init_K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_K = self.layer.K
        K = self.manifold.k.to(device=layer_K.device, dtype=layer_K.dtype)
        self.layer.K = K
        if hasattr(self.layer, "manifold") and hasattr(self.layer.manifold, "K"):
            self.layer.manifold.K = K
        return self.layer(x)
