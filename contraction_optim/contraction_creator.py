import torch
import cupy as cp
from cupyx import cutensor
import platform

PERMANENT_ALPHA = 1.0
PERMANENT_BETA = 0.0
PERMANENT_ALGO_NUMBER = -3

# ALGO_DEFAULT_PATIENT = -6  # NOQA, Uses the more accurate but also more time-consuming performance model
# ALGO_GETT = -4             # NOQA, Choose the GETT algorithm
# ALGO_TGETT = -3            # NOQA, Transpose (A or B) + GETT
# ALGO_TTGT = -2             # NOQA, Transpose-Transpose-GEMM-Transpose (requires additional memory)
# ALGO_DEFAULT = -1          # NOQA, Lets the internal heuristic choose

class contraction_handler:
    def __init__(self, a: torch.tensor, b: torch.tensor, contraction_indices: tuple[list, list], alpha_val: float = PERMANENT_ALPHA, beta_val: float = PERMANENT_BETA, contraction_algorithm=PERMANENT_ALGO_NUMBER, debug = False):
        self.a = a
        self.b = b
        self.contraction_indices = contraction_indices
        self.alpha_val = alpha_val
        self.beta_val = beta_val
        self.contraction_algorithm = contraction_algorithm
        self.debug = debug

    def perform_contraction(self):
        # Get the dimensions of the tensors
        aNoDim = len(self.a.shape)
        bNoDim = len(self.b.shape)

        # Construct the Einstein notation
        einstein_notation = self.construct_einstein_notation(aNoDim, bNoDim, self.contraction_indices)
        if self.debug:
            print(einstein_notation)
        self.set_modes(einstein_notation)
        self.extents = self.set_extents(self.a.size(), self.b.size(), self.mode_a, self.mode_b)

        self.c = self.create_C().astype(cp.float32)

        # Perform the contraction
        output = cutensor.contraction(self.alpha_val, cp.from_dlpack((self.a).detach()), self.mode_a, cp.from_dlpack((self.b).detach()), self.mode_b, self.beta_val, self.c, self.mode_c, algo = self.contraction_algorithm)
        return torch.from_dlpack(output)
    
    def construct_einstein_notation(self, aNoDim: int, bNoDim: int, contraction_indices: tuple[list, list]):
        indices = 'abcdefghijklmnopqrstuvwxyz'
        left = ''
        right = ''
        iterator = 0
        contracted_modes = []

        cleaned_contraction_indices = (self.clean_negative_index_postions(aNoDim, contraction_indices[0]), self.clean_negative_index_postions(bNoDim, contraction_indices[1]))
        
        # Iterate over all dimensions of the first tensor
        for i in range(aNoDim):
            left += indices[iterator]
            if i in cleaned_contraction_indices[0]:
                contracted_modes.append(indices[iterator])
            else:
                right += indices[iterator]
            iterator += 1
        
        # Add the '*' symbol to the left side of the equation
        left += ' * '
        
        # Iterate over all dimensions of the second tensor
        for i in range(bNoDim):
            if i in cleaned_contraction_indices[1]:
                left += contracted_modes.pop(0)
            else:
                left += indices[iterator]
                right += indices[iterator]
                iterator += 1
        
        # Return the Einstein notation
        return left + ' -> ' + right

    def clean_negative_index_postions(self, noOfDims: int, contraction_axes: list):
        # convert the negative indices into the actual positions
        for i in range(len(contraction_axes)):
            if contraction_axes[i] < 0:
                contraction_axes[i] = contraction_axes[i] % noOfDims
        return contraction_axes

    def set_modes(self, con_type) -> None:
        AB = con_type.split("->")[0]
        A = AB.split("*")[0]
        B = AB.split("*")[1]
        self.cqinp  = A + ',' + B
        C = con_type.split("->")[1]

        self.mode_a = tuple([i for i in A.split()[0]])
        self.mode_b = tuple([j for j in B.split()[0]])
        self.mode_c = tuple([k for k in C.split()[0]])
    
    def create_C(self):
        return cp.zeros([self.extents[i] for i in self.mode_c])
    
    def set_extents(self, adim, bdim, mode_a, mode_b) -> dict:
        extent_a = {}
        extent_b = {}

        def populate_extent(extent, mode, dim):
            for i in range(len(mode)):
                extent[mode[i]] = dim[i]
            return extent
        
        n_extent_a = populate_extent(extent_a, mode_a, adim)
        n_extent_b = populate_extent(extent_b, mode_b, bdim)
        if platform.version() < '3.9':
            return {**n_extent_a, **n_extent_b}
        else:
            return n_extent_a | n_extent_b