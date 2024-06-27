import numpy as np
from scipy.ndimage import label, generate_binary_structure, binary_dilation

def suppress_all_but_largest(arr):
    """
    Takes a 3D segmentation array and suppresses all but the largest component.
    
    Parameters:
        arr (ndarray): 3D segmentation array
        
    Returns:
        out_arr (ndarray): Segmentation array with all but the largest component suppressed
    """
    # Generate a 3x3x3 binary structure for use with binary dilation
    struct = generate_binary_structure(3, 1)
    
    # Label the connected components in the array
    labeled_arr, num_components = label(arr)
    
    # Find the size of each component
    component_sizes = np.bincount(labeled_arr.ravel())
    component_sizes[0] = 0 # Exclude background
    
    # Find the index of the largest component
    largest_component_index = component_sizes.argmax()
    
    # Create a mask for all components except the largest
    mask = labeled_arr != largest_component_index
    
    # Dilate the mask to include all voxels adjacent to the suppressed components
    #mask = binary_dilation(mask, structure=struct)
    
    # Create the output array and set all suppressed voxels to 0
    out_arr = np.copy(arr)
    out_arr[mask] = 0
    
    return out_arr