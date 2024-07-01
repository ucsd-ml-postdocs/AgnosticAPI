import os
import nibabel as nib
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pickle
from scipy.ndimage import gaussian_filter

import agnosticapi.server.models.seg3d_model.seg3d_backend.ls_seg3d_utils as ut
from agnosticapi.server.models.seg3d_model import model_files, model_history

def compute_gaussian(tile_size, sigma_scale, dtype=np.float32) -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def seg3d_predict(model_path, image_path):
    model = keras.models.load_model(model_path)
    with open(model_history, "rb") as file_pi:
        history = pickle.load(file_pi)

    gaussian_window = compute_gaussian([128, 128, 96], 1. / 8)
    print(gaussian_window.shape)
    print(gaussian_window.dtype)
    
    gauss_window_multiplier = np.repeat(gaussian_window[:, :, :, np.newaxis], 10, axis=3)
    
    # resampled_image_path = ut.resample_single_volume(image_path)
    resampled_image_path = image_path
    
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

    img_crop = ut.crop_or_pad(nii_img_data, crop_dim, value=(-1024 - mean) / std)
    img_pred = np.zeros(np.concatenate((img_crop.shape, [10])))
    norm_array = np.zeros(np.concatenate((img_crop.shape, [10])))

    patch_step_d1 = 64
    patch_step_d2 = 64
    patch_step_d3 = 48

    for i1 in range(0, img_crop.shape[0] - 127, patch_step_d1):
        for i2 in range(0, img_crop.shape[1] - 127, patch_step_d2):
            for i3 in range(0, img_crop.shape[2] - 95, patch_step_d3):
                patch = img_crop[i1:i1 + 128, i2+i2 + 128, i3 + 96]
                x = np.expand_dims(patch, 0)
                x = np.expand_dims(x, 4)
                logits = model.predict(x, verbose=0)

                prob = tf.nn.softmax(logits, axis=-1)
                prob = np.squeeze(prob)
                prob_weight = prob * gauss_window_multiplier
                prob_weight = np.squeeze(prob_weight)

                img_pred[i1:i1 + 128, i2:i2 + 128, i3:i3 + 96, :] = img_pred[i1:i1 + 128, i2:i2 + 128, i3:i3 + 96, :] + prob_weight
                norm_array[i1:i1 + 128, i2:i2 + 128, i3:i3 + 96] = norm_array[i1:i1 + 128, i2:i2 + 128, i3:i3 + 96] + gauss_window_multiplier

    img_pred_norm = img_pred / norm_array
    labels = tf.argmax(img_pred_norm, axis=-1)
    labels = ut.crop_or_pad(labels, orig_size)

    prob = img_pred_norm
    prob = ut.crop_or_pad(prob, tuple(np.concatenate((orig_size, [10]))), value=(-1024 - mean) / std)
    
    labels_nii = nib.Nifti1Image(labels, nii_img.affine, nii_img.header)

    print('Saved prediction shape: (' + str(labels.shape[0]) + ',' + str(labels.shape[1]) + ',' + str(labels.shape[2]) + ')')
    
    os.system('rm -r tmp')
    return labels
