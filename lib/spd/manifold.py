import torch

from lib.geoopt import SymmetricPositiveDefinite
#from lib.geoopt.manifolds.lorentz import math


class CustomSpd(SymmetricPositiveDefinite):
    def _init__(self):
        super(CustomSpd, self).__init__()        
    
   
