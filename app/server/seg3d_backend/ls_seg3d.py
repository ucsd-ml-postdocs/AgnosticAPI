import os
import nibabel as nib
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm

import app.ls_seg3d_utils as ut   # use this for docker
#import ls_seg3d_utils as ut      # use this for running from the command line

from scipy.ndimage import gaussian_filter
def compute_gaussian(tile_size, sigma_scale, dtype=np.float32) \
        -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(
        gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

import pickle

def ls_seg3d(image_path):
    # path_to_data = 'app/ls_seg3D_laptop/'

    model_name = 'app/models/ls_seg3d_model/m20230623-163203wh500epochs'
    history_name = 'app/models/ls_seg3d_model/h20230623-163203wh500epochs'

    model = keras.models.load_model(model_name)
    with open(history_name, "rb") as file_pi:
        history = pickle.load(file_pi)

    gaussian_window = compute_gaussian([128, 128, 96], 1. / 8)
    print(gaussian_window.shape)
    print(gaussian_window.dtype)

    gauss_window_multiplier = np.repeat(gaussian_window[:, :, :, np.newaxis], 10,
                                        axis=3)
    print(gauss_window_multiplier.shape)

    save_folder = 'app/Predictions/p20230623-163203wh500epochs/'
    os.makedirs(save_folder, exist_ok=True)

    resampled_image_path = ut.resample_single_volume(image_path)

    # load and normalize image
    nii_img = nib.load(resampled_image_path)
    nii_img_data = nii_img.get_fdata()
    # normalize to MEAN AND STD OF POPULATION!
    mean = -79
    std = 411
    nii_img_data = (nii_img_data - mean) / std

    orig_size = nii_img_data.shape

    # crop to easy size
    crop_dim = (256, 256, 240)

    img_crop = ut.crop_or_pad(nii_img_data, crop_dim,
                              value=(-1024 - mean) / std)
    print("img_crop shape: ", img_crop.shape)
    print("nii_img_data shape: ", nii_img_data.shape)
    # preallocate space for prediction
    img_pred = np.zeros(np.concatenate((img_crop.shape, [10])))
    norm_array = np.zeros(np.concatenate((img_crop.shape, [10])))

    # break image into patches and predict
    patch_step_d1 = 64  # 32
    patch_step_d2 = 64  # 32
    patch_step_d3 = 48  # 24

    for i1 in range(0, img_crop.shape[0] - 127, patch_step_d1):
        for i2 in range(0, img_crop.shape[1] - 127, patch_step_d2):
            for i3 in range(0, img_crop.shape[2] - 95, patch_step_d3):
                patch = img_crop[i1:i1 + 128, i2:i2 + 128, i3:i3 + 96]
                print(i1,i2,i3, patch.shape)
                x = np.expand_dims(patch, 0)
                x = np.expand_dims(x, 4)
                logits = model.predict(x,
                                       verbose=0)  # predict on this patch

                prob = tf.nn.softmax(logits,
                                     axis=-1)  # convert to probabilities
                prob = np.squeeze(prob)

                # multiply probabilities by gaussian window
                prob_weight = prob * gauss_window_multiplier

                # remove first dimension
                prob_weight = np.squeeze(prob_weight)

                img_pred[i1:i1 + 128, i2:i2 + 128, i3:i3 + 96,
                :] = img_pred[i1:i1 + 128, i2:i2 + 128, i3:i3 + 96,
                     :] + prob_weight
                norm_array[i1:i1 + 128, i2:i2 + 128,
                i3:i3 + 96] = norm_array[i1:i1 + 128, i2:i2 + 128,
                              i3:i3 + 96] + gauss_window_multiplier

    # normalize prediction by number of patches pixel was predicted on
    img_pred_norm = img_pred / norm_array

    # convert probability to label using argmax
    labels = tf.argmax(img_pred_norm,
                       axis=-1)  # convert from probability to label

    ## if image was larger than crop size, pad to original size
    prob = ut.crop_or_pad(prob, tuple(np.concatenate((orig_size, [10]))),
                          value=(-1024 - mean) / std)
    labels = ut.crop_or_pad(labels, orig_size)

    prob_nii = nib.Nifti1Image(prob, nii_img.affine, nii_img.header)
    img_crop_nii = nib.Nifti1Image(nii_img_data, nii_img.affine,
                                   nii_img.header)
    labels_nii = nib.Nifti1Image(labels, nii_img.affine, nii_img.header)

    # where to save prediction and cropped image/segmentation
    vol_folder = save_folder
    if not os.path.exists(vol_folder):
        # print(vol_folder)
        os.mkdir(vol_folder)
        # os.mkdir(vol_folder+'pred-nii-p20230623-163203wh500epochs/')
        os.mkdir(vol_folder + 'seg-nii-p20230623-163203wh500epochs/')
    
    ### The following code will save the predictions 
    '''
    nib.save(labels_nii, vol_folder+'/seg_'+image_path.split('/')[-1])
    print('Saved prediction shape: (' + str(labels.shape[0]) + ',' + str(
        labels.shape[1]) + ',' + str(labels.shape[0]) + ')')
    '''
    
    os.system('rm -r tmp')
    return labels
