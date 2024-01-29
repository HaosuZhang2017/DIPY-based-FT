# -*- coding: utf-8 -*-
"""
Performs all registrations
This script can be called as script.py /path/to/files/
Here, we expect to find these files:
    1) dti.nii.gz # 4D nifti
    2) t1.nii.gz
    3) tms_points.nii.gz # In T1 space
"""

"""
DEFINE PATH TO YOUR ATLAS HERE
"""

atlas_c = 'mni'

if atlas_c == 'aal':
    atlas_t1 = "/Users/Desktop/Atlas/t1_bet.nii.gz" #This is the T1 (AAL90) corresponding to the atlas (i.e. same space, after skull strip!)
    atlas = "/Users/Desktop/Atlas/aal.nii.gz" #original AAL90
if atlas_c == 'mni':
    atlas_t1 = "/Users/Desktop/Atlas/MNI_bet.nii.gz" #This is the T1 (MNI) corresponding to the atlas (i.e. same space, after skull strip!)
    atlas = "/Users/Desktop/Atlas/MNI.nii" #original MNI
    )
    
gpu = False #Set to true if you have a CUDA-compatible GPU with at least 4GB

import subprocess
import shlex
import nibabel as nib
import sys
import numpy as np
import os

folder = sys.argv[1] + "/"

#Strip b0
dti_file = folder + "/dti.nii.gz"
img_file = nib.load(dti_file)
img = img_file.get_fdata()
b0 = img[:,:,:,0]
nib.save(nib.Nifti1Image(b0.astype(np.float32),img_file.affine,img_file.header),folder+"/b0.nii.gz")

print("Strip b0 done")

#Skullstrip T1 using HD-BET
if not os.path.exists(folder + "/t1_bet.nii.gz"):
    if gpu:
        bet_call = "hd-bet -i " + folder + "/t1.nii.gz -o " + folder + "/t1_bet.nii.gz"
        subprocess.run(shlex.split(bet_call),stdout=subprocess.PIPE,shell=False)
    else:
        bet_call = "hd-bet -i " + folder + "/t1.nii.gz -device cpu -mode fast -tta 0 -o " + folder + "/t1_bet.nii.gz"
        subprocess.run(shlex.split(bet_call),stdout=subprocess.PIPE,shell=False)
    
print("Skullstrip T1 done")

#Register T1 -> DTI and warp tms_points
reg_call_intra = "antsRegistrationSyNQuick.sh -d 3 -m " + folder + "/t1_bet.nii.gz -f " + folder + "/b0.nii.gz -o " + folder + "/intra_reg -t r"
subprocess.run(shlex.split(reg_call_intra),stdout=subprocess.PIPE,shell=False)

# res_call_intra = ("antsApplyTransforms -d 3 -i " + folder + "/t1_WK.nii.gz" +
#                     " -r " + folder + "/b0.nii.gz" +
#                     " -t " + folder + "/intra_reg0GenericAffine.mat" +
#                     " -n NearestNeighbor -o " + folder + "/t1_WK_dti.nii.gz")
# subprocess.run(shlex.split(res_call_intra),stdout=subprocess.PIPE,shell=False)

print("Register T1 -> DTI and warp tms_points done")

#Register Atlas_T1 -> T1 (lin; nlin)
reg_call_inter = "antsRegistrationSyNQuick.sh -d 3 -m " + atlas_t1 + " -f " + folder + "/t1_bet.nii.gz -o " + folder + "inter_reg -t s -j 1"
subprocess.run(shlex.split(reg_call_inter),stdout=subprocess.PIPE,shell=False)

print("Register Atlas_T1 -> T1 done")

#Warp atlas -> DTI (lin;nlin;lin)
res_call_inter = ("antsApplyTransforms -d 3 -i " + atlas +
                   " -r " + folder + "/b0.nii.gz" +
                   " -t " + folder + "/intra_reg0GenericAffine.mat" +
                   " -t " + folder + "/inter_reg1Warp.nii.gz" +
                   " -t " + folder + "/inter_reg0GenericAffine.mat" +
                   " -n NearestNeighbor -o " + folder + "/"+ atlas_c + "_atlas_dti.nii.gz")
subprocess.run(shlex.split(res_call_inter),stdout=subprocess.PIPE,shell=False)

print("Warp atlas -> DTI done")

os.remove(folder + '/intra_regWarped.nii.gz')
os.remove(folder + '/inter_reg1InverseWarp.nii.gz')
os.remove(folder + '/inter_regWarped.nii.gz')
os.remove(folder + '/inter_regInverseWarped.nii.gz')
os.remove(folder + '/intra_regInverseWarped.nii.gz')
os.remove(folder + '/inter_reg1Warp.nii.gz')
os.remove(folder + '/inter_reg0GenericAffine.mat')
os.remove(folder + '/intra_reg0GenericAffine.mat')

print("All registrations done")
