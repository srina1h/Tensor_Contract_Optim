from ctypes import Union
import math
from re import M
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .low_rank_tensors import TensorTrain,TensorTrainMatrix
from .utils import config_class, TT_forward
from .emb_utils import get_cum_prod,tensorized_lookup

# from .tt_fwd_bwd import TT_forward_quant


# from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
# from ..common_types import _size_1_t, _size_2_t, _size_3_t

class wrapped_linear_layers(nn.Module):
    def __init__(self,in_features,out_features,bias=True,tensorized=False,config=None):
        super(wrapped_linear_layers,self).__init__()
        if tensorized==True:
            self.layer = TensorizedLinear_module(in_features,out_features, config, bias=bias)
        else: 
            self.layer = torch.nn.Linear(in_features,out_features,bias=bias)
    
        self.tensorized = tensorized
    def forward(self,input,config_forward=None):
        if self.tensorized:
            return self.layer(input,config_forward=config_forward)
        else:
            return self.layer(input)

class TensorizedEmbedding(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                config):

        super(TensorizedEmbedding,self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.shape = config.shape

        # target_stddev = np.sqrt(1/(np.prod(self.shape[0])+np.prod(self.shape[1])))
        target_stddev = 1.0

        
        config_tensor = config_class(shape=config.shape,ranks=config.ranks,target_sdv=target_stddev)

        self.tensor = TensorTrainMatrix(config_tensor)


        self.cum_prod = get_cum_prod(self.shape)



    def forward(self, x, config_forward=None):

        xshape = list(x.shape)
        xshape_new = xshape + [self.out_features, ]
        # x = x.view(-1)
        x = torch.flatten(x)
        
        factors =self.tensor.factors
        rows = tensorized_lookup(x,factors,self.cum_prod,self.shape,'TensorTrainMatrix')

        rows = rows.view(x.shape[0], -1)

        rows = rows.view(*xshape_new)
        

        return rows.to(x.device)



class TensorizedLinear_module(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                config,
                bias=True
    ):
        """
        config has following attributes:
        shape: the shape of the tensor
        ranks: either a number or a list of numbers to specify the ranks 
        set_scale_factors: True or False
        """
    

        super(TensorizedLinear_module,self).__init__()

        self.config = config

        

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(1/(self.in_features+self.out_features))

        config_tensor = config_class(shape=config.shape,ranks=config.ranks,target_sdv=target_stddev)

        #shape taken care of at input time
        self.tensor = TensorTrain(config_tensor)

        if bias == False:
            self.bias = 0
        else:
            stdv = 1. / math.sqrt(out_features)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)
            



    def forward(self,input,config_forward=None):
        """
        config_forward:
        prune_mask: True or False. Use prune mask or not 
        threshold: float number. The threshold to clip rank_parameters to 0
        quantized: 0: full precision. 1: quantization-aware training. 2: low-precision training.
        if quantized:
            rep: INT or FLOAT. quantization type
            bit_input/factors/intermediate/out: bits for each part
            rounding: stochastic or nearest. Rounding type
        """
      
        factors =self.tensor.factors

        # out = self.forward_tt_full_precision(input,factors) + self.bias
        out = TT_forward.apply(input,*factors)+self.bias


        return out 

    




