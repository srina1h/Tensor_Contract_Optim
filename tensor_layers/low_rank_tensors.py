#%%
import torch 
import math


# class config_tensor:
#     def __init__(self,
#                 shape = None,
#                 ranks = None,
#                 target_sdv = 1.0,
#                 **kwargs):
#         self.shape = shape
#         self.ranks = ranks
#         self.target_sdv = target_sdv
#         for x in kwargs:
#             setattr(self, x, kwargs.get(x))

class TensorTrain(torch.nn.Module):
    def __init__(self,config):

        super(TensorTrain, self).__init__()

        self.config = config
        self.factors = torch.nn.ParameterList()
        self.order = len(config.shape)
        self.build_factors_Gaussian()

    def build_factors_uniform(self):
        config = self.config
        shape = config.shape
        ranks = config.ranks
        order = len(shape)
        if type(ranks)==int:
            ranks = [1]+[ranks]*(order-1)+[1]
        
        for i in range(order):
            n = shape[i]
            r1 = ranks[i]
            r2 = ranks[i+1]
            U = torch.nn.Parameter(torch.randn(r1,n,r2)/math.sqrt(r2)*self.config.target_sdv**(1/order))
            self.factors.append(U)



    def build_factors_Gaussian(self):
        config = self.config
        shape = config.shape
        ranks = config.ranks
        order = self.order
        if type(ranks)==int:
            ranks = [1]+[ranks]*(order-1)+[1]
        
        for i in range(order):
            n = shape[i]
            r1 = ranks[i]
            r2 = ranks[i+1]
            U = torch.nn.Parameter(torch.randn(r1,n,r2)/math.sqrt(r2)*self.config.target_sdv**(1/order))
            self.factors.append(U)


    
    def get_full(self,factors):
        with torch.no_grad():
            out = factors[0]
            for U in factors[1:]:
                out = torch.tensordot(out,U,[[-1],[0]])
            
        return torch.squeeze(out)




class TensorTrainMatrix(torch.nn.Module):
    def __init__(self,config):

        super(TensorTrainMatrix, self).__init__()

        self.config = config
        self.factors = torch.nn.ParameterList()
        self.order = len(config.shape[0])
        self.build_factors_Gaussian()

    def build_factors_uniform(self):
        config = self.config
        shape = config.shape
        ranks = config.ranks
        order = len(shape)
        if type(ranks)==int:
            ranks = [1]+[ranks]*(order-1)+[1]
        
        for i in range(order):
            n = shape[i]
            r1 = ranks[i]
            r2 = ranks[i+1]
            U = torch.nn.Parameter(torch.randn(r1,n,r2)/math.sqrt(r2)*self.config.target_sdv**(1/order))
            self.factors.append(U)


    def build_factors_Gaussian(self):
        config = self.config
        shape = config.shape
        ranks = config.ranks
        order = self.order
        if type(ranks)==int:
            ranks = [1]+[ranks]*(order-1)+[1]
        
        for i in range(order):
            n1 = shape[0][i]
            n2 = shape[1][i]
            r1 = ranks[i]
            r2 = ranks[i+1]
            U = torch.nn.Parameter(torch.randn(r1,n1,n2,r2)/math.sqrt(r2)*self.config.target_sdv**(1/order))
            self.factors.append(U)



    
    def get_full(self,factors):
        with torch.no_grad():
            out = factors[0]
            for U in factors[1:]:
                out = torch.tensordot(out,U,[[-1],[0]])
            
        return torch.squeeze(out)




