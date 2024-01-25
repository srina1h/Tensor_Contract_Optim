import torch
import numpy as np
import torch.nn.functional as F

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
                output = (torch.tensordot(output, core, dims=([-1], [0])))
                left.append(output)
            
        
            output = torch.tensordot(matrix,output,[list(range(1,d+1)),list(range(d))])


            saved_tensors.append(left)

            temp = factors[d]
            right.append(temp)
            for core in factors[d + 1:]:
                temp = (torch.tensordot(temp, core, dims=([-1], [0])))
                right.append(temp)

            out = torch.squeeze(temp)

            output = torch.tensordot(output,out,[[-1],[0]])
            output = torch.reshape(output,out_shape)
        
            
            saved_tensors.append(right)
            ctx.saved_tensors_custom = saved_tensors
       
   
        return output

       
    @staticmethod
    def backward(ctx, dy):
        print("backward")
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

            print("back-tdot1")
            print(dy.shape)
            print(dy_core_prod.shape)
            print(dy_core_prod.reshape(dy_core_prod.shape[0], -1).shape)
            dy_core_prod = (torch.tensordot(dy, dy_core_prod.reshape(dy_core_prod.shape[0], -1), dims=([1], [1])))
            print(dy_core_prod.shape)

            print("back-tdot2")
            print(matrix.shape)
            print(dy_core_prod.shape)
            matrix_dy_core_prod = torch.tensordot(matrix, dy_core_prod, dims=([0], [0]))
            print(matrix_dy_core_prod.shape)

            for i in reversed(range(1, d)):
                print("back-tdot-loop1"+str(i))
                print(left[i - 1].shape)
                print(left[i - 1].reshape(-1, ranks[i]))
                print(matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1, ranks[d]).shape)
                grad = (torch.tensordot(left[i - 1].reshape(-1, ranks[i]),
                                    matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1,
                                                                ranks[d]),
                                    dims=([0], [0])))
                print(grad.shape)
                if i == d - 1:
                    right_core = factors[i]
                else:
                    print("back-tdot-loop-else"+str(i))
                    print(grad.shape)
                    print(right_core.shape)
                    grad = (torch.tensordot(grad, right_core, dims=([2, 3], [1, 2])))
                    print(grad.shape)

                    print(factors[i].shape)
                    print(right_core.shape)
                    right_core = torch.tensordot(factors[i], right_core,
                                                dims=([-1], [0])).reshape(ranks[i], -1, ranks[d])
                    print(right_core.shape)
                if grad.shape != factors[i].shape:
                    grad = grad.reshape(list(factors[i].shape))
                # print(grad.shape)
                left_grads.append(grad)
            
            print("back-tdot3")
            print(matrix_dy_core_prod.shape)
            print(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]))
            print(right_core.shape)
            temp = (torch.tensordot(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]),
                                            right_core, dims=([1, 2], [1, 2])).reshape(1, tt_shape_row[0], -1))
            print(temp.shape)

            left_grads.append(temp)

            left_grads = left_grads[::-1]

            matrix_core_prod = left[-1]
            print("back-tdot4")
            print(matrix_core_prod.shape)
            print(matrix_core_prod.reshape(-1, ranks[d]).shape)
            print(dy.shape)
            matrix_core_prod = (torch.tensordot(matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1]),
                                            matrix, dims=([0], [1])))
            print(matrix_core_prod.shape)
            
            # print('dx=',torch.max(matrix_core_prod))
            print("back-tdot5")
            print(matrix_core_prod.shape)
            print(dy.shape)
            matrix_dy_core_prod = (torch.tensordot(matrix_core_prod, dy, dims=([1], [0])))
            print(matrix_dy_core_prod.shape)

            for i in reversed(range(1, d)):
                print("back-tdot-loop2"+str(i))
                print(right[i - 1].shape)
                print(right[i - 1].reshape(-1, ranks[d + i]))
                print(matrix_dy_core_prod.shape)
                print(matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))))
                grad = (torch.tensordot(right[i - 1].reshape(-1, ranks[d + i]),
                                    matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))),
                                    dims=([0], [0])))
                print(grad.shape)
                if i == d - 1:
                    right_core = factors[d + i].reshape(-1, tt_shape_col[i])
                else:
                    print("back-tdot-loop2-else"+str(i))
                    print(grad.shape)
                    print(right_core.shape)
                    grad = (torch.tensordot(grad, right_core, dims=([-1], [1])))
                    print(grad.shape)

                    print("back-tdot-loop2-else2"+str(i))
                    print(factors[d + i].shape)
                    print(right_core.shape)
                    right_core = (torch.tensordot(factors[d + i], right_core, dims=([-1], [0])).reshape(ranks[d + i],-1))
                    print(right_core.shape)                                                                                                                                             
                if grad.shape != factors[d + i].shape:
                    grad = grad.reshape(list(factors[i].shape))

                right_grads.append(grad)
            print("back-tdot6")
            print(matrix_dy_core_prod.shape)
            print(matrix_dy_core_prod.reshape(-1, ranks[d], tt_shape_col[0]))
            print(right_core.shape)
            temp = (torch.tensordot(matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1),
                                            right_core, dims=([-1], [1])))
            print(temp.shape)
            right_grads.append(temp)

            right_grads = right_grads[::-1]

            dx = factors[-1].reshape(ranks[-2], -1)
            for core in reversed(factors[d:-1]):
                print("back-tdot-loop3")
                print(core.shape)
                print(dx.shape)
                dx = (torch.tensordot(core, dx, dims=([-1], [0])))
                print(dx.shape)

            print("back-tdot7")
            print(dy.shape)
            print(dx.shape)
            print(dx.reshape(-1, np.prod(tt_shape_col)).shape)
            dx = (torch.tensordot(dy, dx.reshape(-1, np.prod(tt_shape_col)), dims=([-1], [-1])))
            print(dx.shape)


            temp = factors[0].reshape(-1, ranks[1])
            for core in factors[1:d]:
                print("back-tdot-loop4")
                print(temp.shape)
                print(core.shape)
                temp = (torch.tensordot(temp, core, dims=([-1], [0])))
                print(temp.shape)
            
            print("back-tdot8")
            print(dx.shape)
            print(temp.shape)
            print(temp.reshape(np.prod(tt_shape_row), -1).shape)
            dx = (torch.tensordot(dx, temp.reshape(np.prod(tt_shape_row), -1), dims=([-1], [-1])))
            print(dx.shape)
            dx = torch.reshape(dx,ctx.input_shape)            

            all_grads = [g for g in left_grads+right_grads]



        return dx, *(all_grads)


