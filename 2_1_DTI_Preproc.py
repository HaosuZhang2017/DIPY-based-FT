# -*- coding: utf-8 -*-
"""
Simple script based on DiPy, which will
1) Affine co-register all images in a 4D series to the reference image (usually first b0)
2) Rotate the B-matrix accordingly
3) Perform a robust tensor estimation using NLLS

Call as script.py "/PATH/TO/FILES/"
/PATH/TO/FILES/ should hold: dti.nii.gz, bvec.bvec, bval.bval
THIS SCRIPT ASSUMES THAT THE FIRST IMAGE IS THE TARGET B0
IN OTHER CASES, PLEASE REWRITE ACCORDINGLY

http://nipy.org/dipy/documentation.html
b.wiestler@tum.de
"""

import sys
import os
import numpy as np
import nibabel as nib

from dipy.align.imaffine import (transform_centers_of_mass,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   AffineTransform3D)
from dipy.core.gradients import (gradient_table,reorient_bvecs)
import dipy.reconst.dti as dti
from dipy.segment.mask import median_otsu
#import dipy.denoise.noise_estimate as ne

"""
Will rotate the B-matrix
"""
def gtab_pp(folder_path,affines):
    """
    Leemans et al., 2009
    
    Parameters:
    -----------
    folder_path: Folder/
    affines: 4D NPArray with the affines

    """
    bval_file = folder_path + "/bval.bval"
    bvec_file = folder_path + "/bvec.bvec"
    
    gtab_old = gradient_table(bval_file,bvec_file)
    
    gtab_cor = reorient_bvecs(gtab_old,affines)
   
    np.savetxt(folder_path+"/bvec_cor.bvec",np.array(gtab_cor.bvecs),delimiter=" ",newline=os.linesep)
    
    print("BVec rotation done")
    
    return gtab_cor
    
"""
Will do the registration
"""
def register_series(series,folder_path):
    """
    Register a series to a reference image.
    
    Parameters
    ----------
    series : Nifti1Image object
       The data is 4D with the last dimension separating different 3D volumes
    folder_path : See below
    
    Returns
    -------
    transformed_list, affine_list
    """
    metric = MutualInformationMetric(nbins = 32, sampling_proportion = 0.25)
    level_iters = [250, 125, 25]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    
    #Assumes that target b0 = 0 (1st volume)
    ref_data = series.get_fdata()[:,:,:,0]
    s_aff = series.affine

    """
    Implement b0-free subsetting of series...
    """
    gdirs = series.get_fdata().shape[3]
    series_data = series.get_fdata()[:,:,:,1:gdirs]
    m_aff = series.affine

    affine_list = []
    transformed_list = []
    for ii in range(series_data.shape[3]):
        this_moving = series_data[:,:,:,ii]
        """
        Employs a multi-refinement approach (like ANTs):
        First, align centers of mass, then use this as starting point for a translation
        Then, use this affine for an affine transform
        """
        c_of_mass = transform_centers_of_mass(ref_data, s_aff,
                                      this_moving, m_aff)
        
        affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)
        
        transform = TranslationTransform3D()
        params0 = None
        starting_affine = c_of_mass.affine
        translation = affreg.optimize(ref_data, this_moving, transform, params0,
                              s_aff, m_aff,
                              starting_affine=starting_affine)
        
        transform = AffineTransform3D()
        params0 = None
        starting_affine = translation.affine
        affine = affreg.optimize(ref_data, this_moving, transform, params0,
                         s_aff,m_aff,
                         starting_affine=starting_affine)
            
        transformed_list.append(affine.transform(this_moving))
        affine_list.append(affine.affine)
        
        print(str((ii+1)) + " / " + str(series_data.shape[3]) + " directions registered")
        
        #For MI:
        #metric.metric_val     

    print("Affine registration on B0 done")
   
    return transformed_list, affine_list

"""
Will calculate tensors
"""
def tensor_estimation(dti_cor,img_affine,img_header,gtab_cor,folder_path):
    """
    Takes as input the co-registered DTI (float64) and estimates tensors
    Parameters
    ----------
    dti_cor: NPArray (float64) with the motion-corrected DTI raw data
    img_affine: Affine image matrix
    gtab_cor: Rotated B-matrix
    folder_path: folder/
    """
    #Brain masking
    b0_data = dti_cor[:,:,:, 0]
    diff_data = dti_cor[:,:,:,1:dti_cor.shape[3]]
    
    b0_masked, mask = median_otsu(b0_data, median_radius = 3, numpass = 2)
    mask_image = nib.Nifti1Image(mask.astype("uint8"),
                                 img_affine,img_header)
    out_path = folder_path + "/median_otsu_mask.nii.gz"
    nib.save(mask_image, out_path)
    
    #Start robust tensor estimation
    #sigma = ne.estimate_sigma(dti_cor)
    dti_restore = dti.TensorModel(gtab_cor,fit_method='NLLS')
    fit_restore = dti_restore.fit(dti_cor,mask)
           
    evecs_img = nib.Nifti1Image(fit_restore.evecs.astype("float32"), img_affine,img_header)
    nib.save(evecs_img, folder_path + "/nlls_V.nii.gz")
    
    evals_img = nib.Nifti1Image(fit_restore.evals.astype("float32"), img_affine,img_header)
    nib.save(evals_img, folder_path + "/nlls_L.nii.gz")
    
    fa = fit_restore.fa
    fa_image = nib.Nifti1Image(fa.astype("float32"),
                               img_affine,img_header)
    out_path = folder_path + "/nlls_FA.nii.gz"
    nib.save(fa_image, out_path)
    
    md = fit_restore.md
    md_image = nib.Nifti1Image(md.astype("float32"),
                               img_affine,img_header)
    out_path = folder_path + "/nlls_MD.nii.gz"
    nib.save(md_image, out_path)
    
    isodiff = np.mean(diff_data[:,:,:,:],axis=3)
    isodiff_image = nib.Nifti1Image(isodiff.astype("float32"),
                               img_affine,img_header)
    out_path = folder_path + "/isoDiff.nii.gz"
    nib.save(isodiff_image, out_path)
    
    print("Tensor estimation done")

def main_reg(folder_path):
    """
    Prepare everything
    Parameters
    ----------
    folder_path: String to the folder/    
    """
    dti_file = folder_path + "/dti.nii.gz"
    img = nib.load(dti_file)
    
    transformed_list, affine_list = register_series(img,folder_path)
    
    #Reorient GTAB
    gtab_cor = gtab_pp(folder_path,affine_list)
    
    # Construct a series out of the ref B0 and the reg. volumes:
    transformed_array = np.array(transformed_list)
    transformed_array = np.transpose(transformed_array,axes=(1,2,3,0))
    b0_data = img.get_fdata()[:,:,:,0]
    transformed_array = np.insert(transformed_array,0,b0_data,3)
    
    tensor_estimation(transformed_array,img.affine,img.header,gtab_cor,folder_path)

    #Save some memory (Convert float64 -> float32)
    reg_series = nib.Nifti1Image(transformed_array.astype("float32"),
                                 img.affine)

    out_path = dti_file.replace(".nii.gz","_cor.nii.gz")
    nib.save(reg_series, out_path)

main_reg(sys.argv[1])