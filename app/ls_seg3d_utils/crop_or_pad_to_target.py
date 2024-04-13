# System

# Third Party
import numpy as np

# Internal


def crop_or_pad(array, target, value=0):
    """
    Symmetrically pad or crop along each dimension to the specified target dimension.
    :param array: Array to be cropped / padded.
    :type array: array-like
    :param target: Target dimension.
    :type target: `int` or array-like of length array.ndim
    :returns: Cropped/padded array. 
    :rtype: array-like
    """
    # Pad each axis to at least the target.
    margin = target - np.array(array.shape)
    padding = [(0, max(x, 0)) for x in margin]
    array = np.pad(array, padding, mode="constant", constant_values=value)
    #array = np.pad(array, padding, mode="symmetric")
    for i, x in enumerate(margin):
        array = np.roll(array, shift=+(x // 2), axis=i)

    if type(target) == int:
        target = [target] * array.ndim

    ind = tuple([slice(0, t) for t in target])
    return array[ind]






import numpy as np

def mirror_center_3d(arr, size):
    """
    Mirrors a 3D NumPy array to a specified size by padding and cropping,
    and mirrors the result along all edges so that the original image is
    centered in the output array.
    
    Parameters:
        arr (ndarray): Input 3D array to be mirrored
        size (tuple): Desired output size (depth, height, width)
        
    Returns:
        mirrored_arr (ndarray): 3D array mirrored to the specified size
    """
    # Determine the amount of padding needed along each axis
    depth_pad = max(size[0] - arr.shape[0], 0)
    height_pad = max(size[1] - arr.shape[1], 0)
    width_pad = max(size[2] - arr.shape[2], 0)
    pad_width = ((depth_pad // 2, depth_pad // 2 + depth_pad % 2),
                 (height_pad // 2, height_pad // 2 + height_pad % 2),
                 (width_pad // 2, width_pad // 2 + width_pad % 2))
    
    # Pad the array with mirrored edges
    padded_arr = np.pad(arr, pad_width, 'reflect')
    
    # Crop the array to the desired size
    depth_crop = min(size[0], padded_arr.shape[0])
    height_crop = min(size[1], padded_arr.shape[1])
    width_crop = min(size[2], padded_arr.shape[2])
    crop_start = ((padded_arr.shape[0] - depth_crop) // 2,
                  (padded_arr.shape[1] - height_crop) // 2,
                  (padded_arr.shape[2] - width_crop) // 2)
    cropped_arr = padded_arr[crop_start[0]:crop_start[0]+depth_crop,
                             crop_start[1]:crop_start[1]+height_crop,
                             crop_start[2]:crop_start[2]+width_crop]
    
    ## Mirror the array along all edges
    #mirrored_arr = cropped_arr.copy()
    #for axis in range(3):
    #    mirrored_arr = np.flip(mirrored_arr, axis=axis)
    
    return cropped_arr


def unmirror_center_3d(mirrored_arr, orig_shape):
    """
    Reverses the mirror_center_3d function and returns the original NumPy array.
    
    Parameters:
        mirrored_arr (ndarray): Mirrored 3D array to be reversed
        orig_shape (tuple): Original shape (depth, height, width) of the array before mirroring
        
    Returns:
        orig_arr (ndarray): Original 3D array
    """
    # Calculate the crop sizes to restore the original array shape
    crop_size = [(mirrored_arr.shape[i] - orig_shape[i]) // 2 for i in range(3)]
    
    # Crop the mirrored array to restore the original array shape
    orig_arr = mirrored_arr[crop_size[0]:orig_shape[0]+crop_size[0], crop_size[1]:orig_shape[1]+crop_size[1],
                            crop_size[2]:orig_shape[2]+crop_size[2]]
    
    return orig_arr



