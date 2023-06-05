from ctypes import Union
import math
from re import M
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .low_rank_tensors import TensorTrain,TensorTrainMatrix
from .utils import config_class, quantize, TT_forward_quant
from .emb_utils import get_cum_prod,tensorized_lookup, TTM_lookup_LP
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

    def set_scale_factors(self,scale_w=1.0):
        self.scales = torch.nn.ParameterList()
        self.scale_factors = torch.nn.ParameterList()

        if not isinstance(scale_w,list):
            scale_w = [scale_w]*len(self.tensor.factors)
        for s in scale_w:
            self.scale_factors.append(torch.nn.Parameter(torch.tensor(s,requires_grad=True)))
        
        self.scale_row = torch.nn.Embedding(self.in_features,1)
        self.scale_row.weight.data[:] = 1.0



    def forward(self, x, config_forward=None):

        xshape = list(x.shape)
        xshape_new = xshape + [self.out_features, ]
        # x = x.view(-1)
        x = torch.flatten(x)

        if config_forward==None:
            factors =self.tensor.get_factors(prune_mask=False)
            rows = tensorized_lookup(x,factors,self.cum_prod,self.shape,'TensorTrainMatrix')
        else:
            factors = self.tensor.get_factors(prune_mask=config_forward.prune_mask,threshold=config_forward.threshold)
            if config_forward.emb_quantized == 0:
                rows = tensorized_lookup(x,factors,self.cum_prod,self.shape,'TensorTrainMatrix')
            elif config_forward.emb_quantized == 1:
                Q_factors = []
                for i,U in enumerate(factors):
                    Q_factors.append(quantize.apply(U,self.scale_factors[i],config_forward.emb_bit_factors,config_forward.emb_rep,config_forward.emb_rounding)) 
                factors = Q_factors
                rows = tensorized_lookup(x,factors,self.cum_prod,self.shape,'TensorTrainMatrix')
            elif config_forward.emb_quantized == 2:
                Q_factors = []
                for i,U in enumerate(factors):
                    Q_factors.append(quantize.apply(U,self.scale_factors[i],config_forward.emb_bit_factors,config_forward.emb_rep,config_forward.emb_rounding)) 

                factors = Q_factors
                # print(torch.max(factors[0]))
                # print(torch.mean(torch.abs(factors[0])))
                
                rows = TTM_lookup_LP(x,factors,self.cum_prod,self.shape,config_forward.emb_rounding,config_forward.emb_bit_factors)
#        rows = gather_rows(self.tensor, x_ind)

        rows = rows.view(x.shape[0], -1)

        rows = rows.view(*xshape_new)
        
        # if config_forward!=None and config_forward.emb_quantized > 0:
            
        #     row_scale = self.scale_row(x).view(*xshape_new[:-1],1)
        #     rows = rows*row_scale

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
            # self.bias = torch.nn.Parameter(torch.zeros(out_features))
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        
        if hasattr(config,'set_scale_factors') and config.set_scale_factors==True:
            self.set_scale_factors()
    
    def set_scale_factors(self,scale_w=1.0,scale_input=1.0,scale_intermediate=1.0,scale_dy=1.0,scale_x=1.0,scale_out=1.0):
        self.scales = torch.nn.ParameterList()
        self.scale_factors = torch.nn.ParameterList()

        if not isinstance(scale_w,list):
            scale_w = [scale_w]*self.tensor.order
        for s in scale_w:
            self.scale_factors.append(torch.nn.Parameter(torch.tensor(s)))

        self.scale_input = torch.nn.Parameter(torch.tensor(scale_input))
        self.scale_intermediate = torch.nn.Parameter(torch.tensor(scale_intermediate))
        self.scale_dy = torch.nn.Parameter(torch.tensor(scale_dy))
        self.scale_x = torch.nn.Parameter(torch.tensor(scale_x))
        self.scale_out = torch.nn.Parameter(torch.tensor(scale_out))


        self.scales.append(self.scale_input)
        self.scales.append(self.scale_intermediate)
        self.scales.append(self.scale_dy)
        self.scales.append(self.scale_x)
        self.scales.append(self.scale_out)


        


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
        if config_forward==None:
            factors =self.tensor.get_factors(prune_mask=False)
        else:
            factors = self.tensor.get_factors(prune_mask=config_forward.prune_mask,threshold=config_forward.threshold)
        

        if config_forward==None or config_forward.quantized==0:
            out = self.forward_tt_full_precision(input,factors) + self.bias

        elif config_forward.quantized == 1:
            out = self.forward_tt_quantization_aware(input,factors,config_forward) + self.bias

        elif config_forward.quantized == 2:
            input = quantize.apply(input,self.scale_input,config_forward.bit_input,config_forward.rep,config_forward.rounding)
            Q_factors = []
            for i,U in enumerate(factors):
                Q_factors.append(quantize.apply(U,self.scale_factors[i],config_forward.bit_factors,config_forward.rep,config_forward.rounding))
            out = TT_forward_quant.apply(config_forward.rounding,config_forward.bit_intermediate,self.scale_intermediate,self.scale_dy,self.scale_dy,input,*Q_factors).clone()

            #need to implement low-precision training codes
            pass
        return out 

    def forward_tt_quantization_aware(self,input,factors,config_forward):
        # input, scale, bit=[1,7,0],rep='INT',rounding='nearest'
        input = quantize.apply(input,self.scale_input,config_forward.bit_input,config_forward.rep,config_forward.rounding)
        Q_factors = []
        for i,U in enumerate(factors):
            Q_factors.append(quantize.apply(U,self.scale_factors[i],config_forward.bit_factors,config_forward.rep,config_forward.rounding))
        factors = Q_factors

        quant_intermediate = lambda x: quantize.apply(x,self.scale_intermediate,config_forward.bit_intermediate,config_forward.rep,config_forward.rounding)

        quant_x = lambda x: quantize.apply(x,self.scale_x,config_forward.bit_intermediate,config_forward.rep,config_forward.rounding)

        quant_out = lambda x: quantize.apply(x,self.scale_out,config_forward.bit_out,config_forward.rep,config_forward.rounding)


        m = len(factors)//2
        N = len(input.shape)
        if len(input.shape)==2:
            mat_shape = [input.shape[0]] + [U.shape[1] for U in factors[0:m]]
        elif len(input.shape)==3:
            mat_shape = [input.shape[0]]+[input.shape[1]] + [U.shape[1] for U in factors[0:m]]
        input = torch.reshape(input, [1] + mat_shape)

      
        out = factors[0]
        
        out = torch.squeeze(out)

        for i in range(1,m):
            U = factors[i]
            out = quant_intermediate(torch.tensordot(out, U, [[-1],[0]]))


        # S = 100
        out = quant_x(torch.tensordot(input, out, [list(range(N,N+m)), list(range(0,m))]))

        out = [out] + list(factors[m:])



        N = len(out[0].shape)
        output = factors[m]


        for i in range(m+1,2*m):
            U = factors[i]
            output = quant_intermediate(torch.tensordot(output,U,[[-1],[0]]))
        
        output = torch.tensordot(out[0],output,[[-1],[0]])
        # output = quant_out(output)

        output = torch.flatten(output, start_dim = N-1, end_dim = -1)
        output = torch.squeeze(output)


        return output

    
    def forward_tt_full_precision(self,input,factors):
      
        m = len(factors)//2
        
 
        if len(input.shape)==2:
            mat_shape = [input.shape[0]] + [U.shape[1] for U in factors[0:m]]
            N=2
        elif len(input.shape)==3:
            N=3
            mat_shape = [input.shape[0]]+[input.shape[1]] + [U.shape[1] for U in factors[0:m]]
        input = torch.reshape(input, [1] + mat_shape)

      
        out = factors[0]
        
        out = torch.squeeze(out)
        output = factors[m]

        for i in range(1,m):
            U = factors[i]
            V = factors[i+m]
            
            out = torch.tensordot(out, U, [[-1],[0]])
            output = torch.tensordot(output,V,[[-1],[0]])

        
        S = 100
        out = torch.tensordot(input/S, out, [list(range(N,N+m)), list(range(0,m))]) * S
    

        N = len(out.shape)
        
        
        output = torch.tensordot(out/S,output,[[-1],[0]])*S

        output = torch.flatten(output, start_dim = N-1, end_dim = -1)
        output = torch.squeeze(output)


        return output




