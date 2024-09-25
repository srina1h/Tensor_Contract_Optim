import torch
import numpy as np
import torch.nn.functional as F
import cupy as cp
# from cupyx import cutensor
from contraction_optim.contraction_creator import contraction_handler
from contraction_optim import contraction_logger

class config_class():
    def __init__(self,
                **kwargs):
        for x in kwargs:
            setattr(self, x, kwargs.get(x))

#mannually implemented Tensor-vector multiplication with backward.
class TT_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, *factors):
        for i in factors:
            i.requires_grad_(False)

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
            output = factors[0].reshape(-1, ranks[1]).requires_grad_(False)
            left.append(output)

            for core in factors[1:d]:
                con = contraction_handler(output, core, ([-1], [0]))
                temp_a = output
                output, conTime, eNotation = con.perform_contraction()
                output = (output)
                contraction_logger.log_arguments(None, None, temp_a.shape, core.shape, output.shape, ([-1], [0]), "forward", conTime, "output", "core", "output", eNotation)
                left.append(output)

            temp_b = output
            con = contraction_handler(matrix, output, [list(range(1,d+1)),list(range(d))])
            output, conTime, eNotation = con.perform_contraction()
            contraction_logger.log_arguments(None, None, matrix.shape, temp_b.shape, output.shape, [list(range(1,d+1)),list(range(d))], "forward", conTime, "matrix", "output", "output", eNotation)

            saved_tensors.append(left)

            temp = factors[d]
            right.append(temp)
            for core in factors[d + 1:]:
                temp_a = temp
                con = contraction_handler(temp, core, ([-1], [0]))
                temp, conTime, eNotation = con.perform_contraction()
                temp = (temp)
                contraction_logger.log_arguments(None, None, temp_a.shape, core.shape, temp.shape, ([-1], [0]), "forward", conTime, "temp", "core", "output", eNotation)
                right.append(temp)

            out = torch.squeeze(temp)

            temp_a = output
            con = contraction_handler(output,out,([-1],[0]))
            output, conTime, eNotation = con.perform_contraction()
            output = (output)
            contraction_logger.log_arguments(None, None, temp_a.shape, out.shape, output.shape, ([-1],[0]), "forward", conTime, "output", "out", "output", eNotation)
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
            temp_b = dy_core_prod
            actual_b = dy_core_prod.reshape(dy_core_prod.shape[0], -1)
            con = contraction_handler(dy, dy_core_prod.reshape(dy_core_prod.shape[0], -1), ([1], [1]))
            dy_core_prod, conTime, eNotation = con.perform_contraction()
            dy_core_prod = (dy_core_prod)   
            contraction_logger.log_arguments(None, temp_b.shape, dy.shape, actual_b.shape, dy_core_prod.shape, ([1], [1]), "backward", conTime, "dy", "dy_core_prod-reshape", "dy_core_prod", eNotation)

            con = contraction_handler(matrix, dy_core_prod, ([0], [0]))
            matrix_dy_core_prod, conTime, eNotation = con.perform_contraction()
            contraction_logger.log_arguments(None, None, matrix.shape, dy_core_prod.shape, matrix_dy_core_prod.shape, ([0], [0]), "backward", conTime, "matrix", "dy_core_prod", "matrix_dy_core_prod", eNotation)

            for i in reversed(range(1, d)):
                temp_a = left[i-1]
                temp_b = matrix_dy_core_prod
                actual_a = left[i-1].reshape(-1, ranks[i])
                actual_b = matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1,ranks[d])
                con = contraction_handler(left[i - 1].reshape(-1, ranks[i]),
                                    matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1, ranks[d]), ([0], [0]))
                grad, conTime, eNotation = con.perform_contraction()
                grad = (grad)
                contraction_logger.log_arguments(temp_a.shape, temp_b.shape, actual_a.shape, actual_b.shape, grad.shape, ([0], [0]), "backward", conTime, "left[i-1]-reshape", "matrix_dy_core_prod-reshape", "grad", eNotation)

                if i == d - 1:
                    right_core = factors[i]
                else:
                    temp_a = grad
                    con = contraction_handler(grad, right_core, ([2, 3], [1, 2]))
                    grad, conTime, eNotation = con.perform_contraction()
                    grad = (grad)
                    contraction_logger.log_arguments(None, None, temp_a.shape, right_core.shape, grad.shape, ([2, 3], [1, 2]), "backward", conTime, "grad", "right_core", "grad", eNotation)

                    temp_b = right_core
                    con = contraction_handler(factors[i], right_core, ([-1], [0]))
                    right_core, conTime, eNotation = con.perform_contraction()
                    right_core = right_core.reshape(ranks[i], -1, ranks[d])
                    contraction_logger.log_arguments(None, None, factors[i].shape, temp_b.shape, right_core.shape, ([-1], [0]), "backward", conTime, "factors[i]", "right_core", "right_core", eNotation)
                
                if grad.shape != factors[i].shape:
                    grad = grad.reshape(list(factors[i].shape))

                left_grads.append(grad)

            temp_a = matrix_dy_core_prod
            con = contraction_handler(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]), right_core, ([1, 2], [1, 2]))
            temp, conTime, eNotation = con.perform_contraction()
            temp = (temp.reshape(1, tt_shape_row[0], -1))
            contraction_logger.log_arguments(temp_a.shape, None, matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]).shape, right_core.shape, temp.shape, ([1, 2], [1, 2]), "backward", conTime, "matrix_dy_core_prod-reshape", "right_core", "temp", eNotation)

            left_grads.append(temp)
            left_grads = left_grads[::-1]
            matrix_core_prod = left[-1]

            temp_a = matrix_core_prod
            actual_a = matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1])
            con = contraction_handler(matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1]), matrix, ([0], [1]))
            matrix_core_prod, conTime, eNotation = con.perform_contraction()
            matrix_core_prod = (matrix_core_prod)
            contraction_logger.log_arguments(temp_a.shape, None, actual_a.shape, matrix.shape, matrix_core_prod.shape, ([0], [1]), "backward", conTime, "matrix_core_prod-reshape", "matrix", "matrix_core_prod", eNotation)

            con = contraction_handler(matrix_core_prod, dy, ([1], [0]))
            matrix_dy_core_prod, conTime, eNotation = con.perform_contraction()
            matrix_dy_core_prod = (matrix_dy_core_prod)
            contraction_logger.log_arguments(None, None, matrix_core_prod.shape, dy.shape, matrix_dy_core_prod.shape, ([1], [0]), "backward", conTime, "matrix_core_prod", "dy", "matrix_dy_core_prod", eNotation)

            for i in reversed(range(1, d)):
                con = contraction_handler(right[i - 1].reshape(-1, ranks[d + i]),matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))), ([0], [0]))
                grad, conTime, eNotation = con.perform_contraction()
                grad = (grad)
                contraction_logger.log_arguments(None, None, right[i - 1].reshape(-1, ranks[d + i]).shape, matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))).shape, grad.shape, ([0], [0]), "backward", conTime, "right[i-1]-reshape", "matrix_dy_core_prod-reshape", "grad", eNotation)

                if i == d - 1:
                    right_core = factors[d + i].reshape(-1, tt_shape_col[i])
                else:
                    actual_a = grad
                    con = contraction_handler(grad, right_core, ([-1], [1]))
                    grad, conTime, eNotation = con.perform_contraction()
                    grad = (grad)
                    contraction_logger.log_arguments(None, None, actual_a.shape, right_core.shape, grad.shape, ([-1], [1]), "backward", conTime, "grad", "right_core", "grad", eNotation)

                    actual_b = right_core
                    con = contraction_handler(factors[d + i], right_core, ([-1], [0]))
                    right_core, conTime, eNotation = con.perform_contraction()
                    right_core = (right_core.reshape(ranks[d + i],-1))
                    contraction_logger.log_arguments(None, None, factors[d + i].shape, actual_b.shape, right_core.shape, ([-1], [0]), "backward", conTime, "factors[d+i]", "right_core","right_core", eNotation)                                                                                                                                                                     
                if grad.shape != factors[d + i].shape:
                    grad = grad.reshape(list(factors[i].shape))

                right_grads.append(grad)

            con = contraction_handler(matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1), right_core, ([-1], [1]))
            temp, conTime, eNotation = con.perform_contraction()
            temp = (temp)
            contraction_logger.log_arguments(None, None, matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1).shape, right_core.shape, temp.shape, ([-1], [1]), "backward", conTime, "matrix_dy_core_prod-reshape", "right_core", "temp", eNotation)

            right_grads.append(temp)
            right_grads = right_grads[::-1]
            dx = factors[-1].reshape(ranks[-2], -1)
            for core in reversed(factors[d:-1]):
                actual_b = dx
                con = contraction_handler(core, dx, ([-1], [0]))
                dx, conTime, eNotation = con.perform_contraction()
                dx = (dx)
                contraction_logger.log_arguments(None, None, core.shape, actual_b.shape, dx.shape, ([-1], [0]), "backward", conTime, "core", "dx", "dx", eNotation)

            temp_b = dx
            actual_b = dx.reshape(-1, np.prod(tt_shape_col))
            con = contraction_handler(dy, dx.reshape(-1, np.prod(tt_shape_col)), ([-1], [-1]))
            dx, conTime, eNotation = con.perform_contraction()
            dx = (dx)
            contraction_logger.log_arguments(None, temp_b.shape, dy.shape, actual_b.shape, dx.shape, ([-1], [-1]), "backward", conTime, "dy", "dx-reshape", "dx", eNotation)

            temp = factors[0].reshape(-1, ranks[1])
            for core in factors[1:d]:
                actual_a = temp
                con = contraction_handler(temp, core, ([-1], [0]))
                temp, conTime, eNotation = con.perform_contraction()
                temp = (temp)
                contraction_logger.log_arguments(None, None, actual_a.shape, core.shape, temp.shape, ([-1], [0]), "backward", conTime, "temp", "core", "temp", eNotation)

            actual_a = dx
            temp_b = temp
            actual_b = temp.reshape(np.prod(tt_shape_row), -1)
            con = contraction_handler(dx, temp.reshape(np.prod(tt_shape_row), -1), ([-1], [-1]))
            dx, conTime, eNotation = con.perform_contraction()
            dx = (dx)
            contraction_logger.log_arguments(None, temp_b.shape, actual_a.shape, actual_b.shape, dx.shape, ([-1], [-1]), "backward", conTime, "dx", "temp-reshape", "dx", eNotation)
            dx = torch.reshape(dx,ctx.input_shape)            
            all_grads = [g for g in left_grads+right_grads]
        return dx, *(all_grads)