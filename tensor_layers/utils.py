import torch
import numpy as np
import torch.nn.functional as F
import cupy as cp
from cupyx import cutensor
from contraction_optim.contraction_creator import contraction_handler

class config_class():
    def __init__(self,
                **kwargs):
        for x in kwargs:
            setattr(self, x, kwargs.get(x))

#mannually implemented Tensor-vector multiplication with backward.
class TT_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, *factors):

        with torch.no_grad():

            tt_shape = [U.shape[1] for U in factors]
            ndims = len(factors)
            d = int(ndims / 2)

            ctx.input_shape = matrix.shape
            if len(matrix.shape)==3:
                out_shape = [matrix.shape[0],matrix.shape[1],np.prod(list(tt_shape[d:]))]
                matrix = torch.flatten(matrix,start_dim=0,end_dim=1)
            else:
                out_shape = [matrix.shape[0],np.prod(list(tt_shape[d:]))]
            ctx.out_shape = out_shape

            ctx.factors = factors
            ctx.matrix = matrix


            
    
            ndims = len(factors)
            d = int(ndims / 2)
            ranks = [U.shape[0] for U in factors] + [1]
            tt_shape = [U.shape[1] for U in factors]
            tt_shape_row = list(tt_shape[:d])
            tt_shape_col = list(tt_shape[d:])
            matrix_cols = matrix.shape[0]

            saved_tensors = [matrix]
            left = []
            right = []
            
            matrix = torch.reshape(matrix,[matrix.shape[0]]+tt_shape_row)

            output = factors[0].reshape(-1, ranks[1])
            left.append(output)

            for core in factors[1:d]:
                # output = (torch.tensordot(output, core, dims=([-1], [0])))
                print(output.requires_grad)
                con = contraction_handler(output, core, ([-1], [0]))
                output = (con.perform_contraction())
                left.append(output)
            
        
            # output = torch.tensordot(matrix,output,[list(range(1,d+1)),list(range(d))])
            con = contraction_handler(matrix, output, [list(range(1,d+1)),list(range(d))])
            output = con.perform_contraction()


            saved_tensors.append(left)

            temp = factors[d]
            right.append(temp)
            for core in factors[d + 1:]:
                # temp = (torch.tensordot(temp, core, dims=([-1], [0])))
                con = contraction_handler(temp, core, ([-1], [0]))
                temp = (con.perform_contraction())
                right.append(temp)

            out = torch.squeeze(temp)

            
            
            # output = torch.tensordot(output,out,[[-1],[0]])
            con = contraction_handler(output,out,([-1],[0]))
            output = (con.perform_contraction())
            output = torch.reshape(output,out_shape)
        
            
            saved_tensors.append(right)
            ctx.saved_tensors_custom = saved_tensors
       
   
        return output

       
    @staticmethod
    def backward(ctx, dy):
        with torch.no_grad():
            factors = ctx.factors
            ndims = len(factors)
            d = int(ndims / 2)
            ranks = [U.shape[0] for U in factors] + [1]
            tt_shape = [U.shape[1] for U in factors]
            tt_shape_row = list(tt_shape[:d])
            tt_shape_col = list(tt_shape[d:])
            saved_tensors = ctx.saved_tensors_custom

            
            
            if len(dy.shape)==3:
                dy = torch.flatten(dy,start_dim=0,end_dim=1)


            matrix = saved_tensors[0]
            left = saved_tensors[1]
            right = saved_tensors[2]
            left_grads = []
            right_grads = []

            dy_core_prod = right[-1]


        
            # dy_core_prod = (torch.tensordot(dy, dy_core_prod.reshape(dy_core_prod.shape[0], -1), dims=([1], [1])))
            con = contraction_handler(dy, dy_core_prod.reshape(dy_core_prod.shape[0], -1), ([1], [1]))
            dy_core_prod = (con.perform_contraction())

            # matrix_dy_core_prod = torch.tensordot(matrix, dy_core_prod, dims=([0], [0]))
            con = contraction_handler(matrix, dy_core_prod, ([0], [0]))
            matrix_dy_core_prod = con.perform_contraction()


            for i in reversed(range(1, d)):
                # grad = (torch.tensordot(left[i - 1].reshape(-1, ranks[i]), matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1,ranks[d]),dims=([0], [0])))
                con = contraction_handler(left[i - 1].reshape(-1, ranks[i]),
                                    matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1, ranks[d]), ([0], [0]))
                grad = (con.perform_contraction())
                # print(grad.shape)
                if i == d - 1:
                    right_core = factors[i]
                else:
                    # grad = (torch.tensordot(grad, right_core, dims=([2, 3], [1, 2])))
                    con = contraction_handler(grad, right_core, ([2, 3], [1, 2]))
                    grad = (con.perform_contraction())

                    # right_core = torch.tensordot(factors[i], right_core,dims=([-1], [0])).reshape(ranks[i], -1, ranks[d])
                    con = contraction_handler(factors[i], right_core, ([-1], [0]))
                    right_core = con.perform_contraction().reshape(ranks[i], -1, ranks[d])
                
                if grad.shape != factors[i].shape:
                    grad = grad.reshape(list(factors[i].shape))
                # print(grad.shape)
                left_grads.append(grad)
            # temp = (torch.tensordot(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]), right_core, dims=([1, 2], [1, 2])).reshape(1, tt_shape_row[0], -1))
            con = contraction_handler(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]), right_core, ([1, 2], [1, 2]))
            temp = (con.perform_contraction().reshape(1, tt_shape_row[0], -1))


            left_grads.append(temp)

            left_grads = left_grads[::-1]

            matrix_core_prod = left[-1]
            # matrix_core_prod = (torch.tensordot(matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1]), matrix, dims=([0], [1])))
            con = contraction_handler(matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1]), matrix, ([0], [1]))
            matrix_core_prod = (con.perform_contraction())
            
            # print('dx=',torch.max(matrix_core_prod))
            # matrix_dy_core_prod = (torch.tensordot(matrix_core_prod, dy, dims=([1], [0])))
            con = contraction_handler(matrix_core_prod, dy, ([1], [0]))
            matrix_dy_core_prod = (con.perform_contraction())


            for i in reversed(range(1, d)):
                # grad = (torch.tensordot(right[i - 1].reshape(-1, ranks[d + i]),matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))),dims=([0], [0])))
                con = contraction_handler(right[i - 1].reshape(-1, ranks[d + i]),matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))), ([0], [0]))
                grad = (con.perform_contraction())

                if i == d - 1:
                    right_core = factors[d + i].reshape(-1, tt_shape_col[i])
                else:
                
                    # grad = (torch.tensordot(grad, right_core, dims=([-1], [1])))
                    con = contraction_handler(grad, right_core, ([-1], [1]))
                    grad = (con.perform_contraction())


                    # right_core = (torch.tensordot(factors[d + i], right_core, dims=([-1], [0])).reshape(ranks[d + i],-1))
                    con = contraction_handler(factors[d + i], right_core, ([-1], [0]))
                    right_core = (con.perform_contraction().reshape(ranks[d + i],-1))
                                                                                                                                                                            
                if grad.shape != factors[d + i].shape:
                    grad = grad.reshape(list(factors[i].shape))

                right_grads.append(grad)

            # temp = (torch.tensordot(matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1), right_core, dims=([-1], [1])))
            con = contraction_handler(matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1), right_core, ([-1], [1]))
            temp = (con.perform_contraction())

            right_grads.append(temp)

            right_grads = right_grads[::-1]

            dx = factors[-1].reshape(ranks[-2], -1)
            for core in reversed(factors[d:-1]):
                # dx = (torch.tensordot(core, dx, dims=([-1], [0])))
                con = contraction_handler(core, dx, ([-1], [0]))
                dx = (con.perform_contraction())

        
            # dx = (torch.tensordot(dy, dx.reshape(-1, np.prod(tt_shape_col)), dims=([-1], [-1])))
            con = contraction_handler(dy, dx.reshape(-1, np.prod(tt_shape_col)), ([-1], [-1]))
            dx = (con.perform_contraction())



            temp = factors[0].reshape(-1, ranks[1])
            for core in factors[1:d]:
                # temp = (torch.tensordot(temp, core, dims=([-1], [0])))
                con = contraction_handler(temp, core, ([-1], [0]))
                temp = (con.perform_contraction())


            # dx = (torch.tensordot(dx, temp.reshape(np.prod(tt_shape_row), -1), dims=([-1], [-1])))
            con = contraction_handler(dx, temp.reshape(np.prod(tt_shape_row), -1), ([-1], [-1]))
            dx = (con.perform_contraction())
            dx = torch.reshape(dx,ctx.input_shape)            

            all_grads = [g for g in left_grads+right_grads]



        return dx, *(all_grads)


