import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

import math

from lib.spd.manifold import CustomSpd

class SpdWrappedNormal(torch.nn.Module):
    """ Implementation of a Spd Wrapped Normal distribution with diagonal covariance matrix defined by Nagano et al. (2019).

    - Implemented for use in VAE training.
    - Original source: https://github.com/pfnet-research/hyperbolic_wrapped_distribution
    
    Args:
        mean_H: Mean in SPD manifold (can be batched)
        var: Diagonal of covariance matrix (can be batched)
    """
    def __init__(self, manifold: CustomSpd):
        super(SpdWrappedNormal, self).__init__()

        # Save variables
        self.manifold = manifold  

        self.batch_size = 100  

        self.dim = 5
        identity = torch.eye(self.dim, dtype=torch.float)
        #identity = identity.unsqueeze(0).repeat(self.num_subseqs, 1, 1)
        self.register_buffer('identity', identity)

    def rsample(self, mean_H, covar, num_samples=1, keepdim=False, ret_uv=False):
        """ Implements sampling from Wrapped normal distribution using reparametrization trick.

        Some intermediate results are saved to object for efficient log_prob calculation.

        Returns:
            Returns num_samples points for each gaussian (or batch instance)
            -> If num_samples==1: Returns shape (bs x dim x dim)
            -> If num_samples>1: Returns shape (num_samples x bs x dim x dim)
        """

        # "1. Sample a vector v_t from the Gaussian distribution N(0,Sigma) defined over R^n" (Nagano et al., 2019)
        vT = MultivariateNormal(
                        torch.zeros((mean_H.shape[0], covar.shape[1]), device=covar.device),    # [100, 15] 
                        covar                                                                   # [100, 15, 15]
                    ).rsample((num_samples,))                                                   # [num_samples, 100, 15]

        # 2. Interpret vT as an element of tangent space T_(mu_0)Sym_++^n (Nagano et al., 2019)
        #v = F.pad(vT, pad=(1, 0))        
        v = torch.zeros_like(torch.empty(num_samples, self.batch_size, mean_H.shape[1], mean_H.shape[1]), requires_grad=False)
        tril_ind = torch.tril_indices(row=mean_H.shape[1], col=mean_H.shape[1], offset=0)
        v[..., tril_ind[0], tril_ind[1]] = vT 
        v = 0.5 * (v + torch.transpose(v, -1, -2))

        # 3. Parallel transport the vector v from origin to mu in T_(mu)H^n subspace of R^(n+1) along the geodesic from origin to mu
        u = self.manifold.transp(self.identity, mean_H, v)
        
        #  4. Map u to z by exponential map
        z = self.manifold.expmap(mean_H, u)                                                     # [num_samples, 100, 5, 5]

        if (num_samples==1) and (not keepdim):
            z = z.squeeze(0)

        if ret_uv:
            return z, u, v
        else:
            return z    
