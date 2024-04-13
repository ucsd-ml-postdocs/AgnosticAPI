# python script to resample a folder of scans

import os

def resample(image_path):

    print('hello world')
    
    # get list of patients in image_path
    pat_list = os.listdir(image_path)
    pat_list = [item for item in pat_list if not item.startswith('.')]
    print(pat_list)
    
    path = image_path
    interp = '-interpolation Cubic'
    resamp = '-resample-mm 1.0x1.0x1.0mm'
    
    # loop through scans
    for s in range(0,len(pat_list)):
        print(path+pat_list[s])
        
        # if patient doesn't exist, tell user
        if not os.path.exists(path+pat_list[s]):
            print("No image folder")
            
        else:
            vol_list = os.listdir(path+pat_list[s]+'/img-nii/')
            vol_list = [item for item in vol_list if not item.startswith('.')]
            print(vol_list)
        
        
        
        
        
            # loop through volumes (timeframes)
            for v in range(0,len(vol_list)):
                vol = vol_list[v]
            
                input_file = path+pat_list[s]+'/img-nii/'+vol
                output_folder = path+pat_list[s]+'/img-nii-1.0/'
                output_file = path+pat_list[s]+'/img-nii-1.0/'+vol
            
                if os.path.exists(output_file):
                    print('already done this file')
                else:
                    # if img-nii-1.0 folder doesn't exist, make it to save files
                    if not os.path.exists(output_folder):
                        os.mkdir(output_folder)
                    command = './c3d'+' '+input_file+' '+interp+' '+resamp+' '+path+pat_list[s]+'/img-nii-1.0/'+vol
                    os.system(command)
          
        
            #command = './c3d'+' '+input_file+' '+interp+' '+resamp+' '+path+pat_list[s]+'/img-nii-1.0/'+vol
        
            #os.system(command)

        
            print('')
    #command = './c3d'+' '+path+'img-nii/04_SSF.nii.gz'+' '+interp+' '+resamp+' '+path+'img-nii-1.0/04_SSF.nii.gz'

    #os.system(command)