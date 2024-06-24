#from utils.generate_random_transform import generate_random_transform
#from utils.transform_full_matrix_offset_center import transform_full_matrix_offset_center
#from utils.generate_random_transform import generate_random_transform
from . import *


def random_transform(params, x, y):
    
    translation,rotation,scale,transform_matrix_raw= generate_random_transform(
        params, x.shape[:-1]
    )
    
    transform_matrix = transform_full_matrix_offset_center(
        transform_matrix_raw, x.shape[:-1]
    )
    x = apply_affine_transform_channelwise(
        x,
        transform_matrix[:-1, :],
        channel_index=params.img_channel_index,
        fill_mode=params.fill_mode,
        cval=params.cval,
    )
    
    # For y, mask data, fill mode constant, cval = 0
    y = apply_affine_transform_channelwise(
        y,
        transform_matrix[:-1, :],
        channel_index=params.img_channel_index,
        fill_mode=params.fill_mode,
        cval=params.cval,
    )
        
    return x, y,translation,rotation,scale,transform_matrix

#def zc_random_transform(params,x):
#    translation,rotation,scale,matrix_raw=dv.zc_random(self.augmentation_params,x.shape[:-1])
#    matrix_final=dv.transform_full_matrix_offset_center(matrix_raw,x.shape[:-1])
#    x_aug = dv.apply_affine_transform_channelwise(x,matrix_final[:-1, :], 
#            channel_index=self.augmentation_params.img_channel_index,
#            fill_mode=self.augmentation_params.fill_mode,
#            cval=self.augmentation_params.cval)
#    return x_aug,translation,rotation,scale,matrix_raw,matrix_final

