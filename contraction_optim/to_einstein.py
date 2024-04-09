def construct_einstein_notation(aNoDim: int, bNoDim: int, contraction_indices: tuple[list, list]):
    indices = 'abcdefghijklmnopqrstuvwxyz'
    left = ''
    right = ''
    iterator = 0
    contracted_modes = []

    cleaned_contraction_indices = (clean_negative_index_postions(aNoDim, contraction_indices[0]), clean_negative_index_postions(bNoDim, contraction_indices[1]))
    
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

def clean_negative_index_postions(noOfDims: int, contraction_axes: list):
    # convert the negative indices into the actual positions
    for i in range(len(contraction_axes)):
        if contraction_axes[i] < 0:
            contraction_axes[i] = contraction_axes[i] % noOfDims
    return contraction_axes