import torch
import torch.nn as nn

from lib.geoopt.manifolds.stereographic.math import arsinh, artanh

class Layer(nn.Module):
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

class UnidirectionalPoincareMLR(Layer):
    """ MLR in the Poincare model by Shimizu et al. (2020)
    
        - Source: https://github.com/mil-tokyo/hyperbolic_nn_plusplus
    """
    __constants__ = ['feat_dim', 'num_outcome']

    def __init__(self, feat_dim, num_outcome, bias=True, ball=None):
        super(UnidirectionalPoincareMLR, self).__init__()
        self.ball = ball
        self.feat_dim = feat_dim    
        self.num_outcome = num_outcome
        weight = torch.empty(feat_dim, num_outcome).normal_( 
            mean=0, std=(self.feat_dim) ** -0.5 / self.ball.c.data.sqrt())
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(num_outcome), requires_grad=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return unidirectional_poincare_mlr(
            x, self.weight_g, self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), self.bias, self.ball.c)
    
@torch.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    # parameters
    rc = c.sqrt()
    drcr = 2. * rc * r

    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)

    return 2 * z_norm / rc * arsinh(
        (2. * torch.matmul(rcx, z_unit) * drcr.cosh() - (1. + cx2) * drcr.sinh()) 
        / torch.clamp_min(1. - cx2, 1e-15))

class BusemannPoincareMLR(Layer):
    """
    Hyperbolic multiclass logistic regression layer based on Busemann functions.
    """

    def __init__(self, feat_dim, num_outcome, ball):
        super(BusemannPoincareMLR, self).__init__()
        self.ball = ball
        self.feat_dim = feat_dim
        self.num_outcome = num_outcome    
        
        INIT_EPS = 1e-3
        self.point = nn.Parameter(torch.randn(self.num_outcome, self.feat_dim) * INIT_EPS)
        self.tangent = nn.Parameter(torch.randn(self.num_outcome, self.feat_dim) * INIT_EPS)

    def forward(self, input):                    
        input = torch.reshape(input, (-1, self.feat_dim))
        point = self.ball.expmap0(self.point)    
        distances = torch.zeros_like(torch.empty(input.shape[0], self.num_outcome), device=input.device, requires_grad=False)
        for i in range(self.num_outcome):
            point_i = point[i]
            tangent_i = self.tangent[i] 
            
            distances[:, i] = self.ball.dist2planebm(      
                x=input, a=tangent_i, p=point_i
            )
        return distances 
    


class PoincareMLR(Layer):
    """
    Hyperbolic multiclass logistic regression (PoincareMLR) per Ganea et al. (2018), Eq. 25.

    - Uses geoopt's signed point-to-hyperplane geodesic distance on the Poincaré ball.
    - Applies per-class scaling factor (lambda_p ||a||) / sqrt(c), where
      lambda_p is the conformal factor at the hyperplane point p.
    - For memory efficiency, evaluates distances in a loop over classes to avoid
      forming the large intermediate tensor from -p_k ⊕ x for all classes at once.

    Reference: Ganea et al., "Hyperbolic Neural Networks" (NeurIPS 2018), Eq. 25.
    """

    def __init__(self, feat_dim, num_outcome, ball):
        super().__init__()
        self.ball = ball
        self.feat_dim = feat_dim
        self.num_outcome = num_outcome    
        
        INIT_EPS = 1e-3
        self.point = nn.Parameter(torch.randn(self.num_outcome, self.feat_dim) * INIT_EPS)
        self.tangent = nn.Parameter(torch.randn(self.num_outcome, self.feat_dim) * INIT_EPS)

    def forward(self, input):
        # input: [batch_size, feat_dim] in Poincaré ball coordinates
        x = torch.reshape(input, (-1, self.feat_dim))
        # map learnable points from tangent at 0 to manifold
        p = self.ball.expmap0(self.point)  # [num_outcome, feat_dim]

        # per-class distance in a loop to avoid memory explosion
        logits = torch.zeros(self.num_outcome, device=input.device, dtype=input.dtype, requires_grad=False)
        for i in range(self.num_outcome):
            logits[:, i] = self.ball.dist2plane(
                x=x, p=p[i], a=self.tangent[i], signed=True, scaled=True
            )

        return logits
