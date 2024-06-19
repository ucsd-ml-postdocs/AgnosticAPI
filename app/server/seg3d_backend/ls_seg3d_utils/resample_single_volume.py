# python script to resample a single 3D volume

import os
import sys

'''
@:return the tmp c3d output file path 
'''
def resample_single_volume(path):

    interp = '-interpolation Cubic'
    resamp = '-resample-mm 1.0x1.0x1.0mm'
    
    
    input_file = path
    output_folder = 'tmp/c3d_out/img-nii-1.0/'
    os.makedirs(output_folder, exist_ok=True)
    output_file = f"tmp/c3d_out/img-nii-1.0/{path.split('/')[-1]}"
    print(path.split('/'))

    # if img-nii-1.0 folder doesn't exist, make it to save files
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # print(os.getcwd())
    # print(os.listdir())
    
    #### Joe added code 
    system = sys.platform
    print("Current working directory: ", os.getcwd())
    if system == 'linux':
        command = './app/server/seg3d_backend/c3d_linux'+' '+input_file+' '+interp+' '+resamp+' '+output_file
    elif system == 'darwin':
        command = './app/server/seg3d_backend/c3d_macos_arm' + ' ' + input_file + ' ' + interp + ' ' + resamp + ' ' + output_file
    
    ret = os.system(command)
    # print(f"command: {command}")
    # print(os.listdir())
    if ret == 0:
        print('Successfully resampled this volume')
        return output_file
    else:
        raise SystemError('c3d resampled failed!')
    
 
