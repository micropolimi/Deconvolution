import numpy as np
from skimage import io
from skimage.external import tifffile as tif
import matplotlib.pyplot as plt
import os

from Functions import deconvolution_algorithm, backfree, norm, pad





''' USER PARAMETERS '''
ITERATIONS = 11       # number of iterations of the Richardson-Lucy algorithm
#TVreg_lambda = 2.5    # put a value if you want to use it as the regularization parameter for the TV regularization, put 'noTV' otherwise
THRESHOLD = 200       # threshold for the background removal function

file_name = "cell"

processing_unit = "GPU"    # GPU or CPU





path = "samples"
full_name = os.path.join(path, file_name)

im = io.imread(full_name + '.tif')
imarray = np.array(im, dtype=np.float32)
number_of_dimensions = len(imarray.shape)    # if 4D: ZCXY is the dimension order (C: channels)
print(imarray.shape)
PSF = io.imread('psf_rescaled.tif')
PSFarray = np.array(PSF, dtype=np.float32)

IMG_PLANE = int((imarray.shape[0])/2)
PSF_PLANE = int((PSFarray.shape[0])/2)

if number_of_dimensions == 2:
    PSFarray = PSFarray[PSF_PLANE,:,:]        # for 2D images
elif imarray.shape[0] < PSFarray.shape[0]:    # for little 3D images
    PSFarray = PSFarray[(PSF_PLANE-1, PSF_PLANE, PSF_PLANE+1),:,:]

# PSF elaboration
backPSF = backfree(PSFarray, THRESHOLD)
normPSF = norm(backPSF)
print(normPSF.shape)

if number_of_dimensions == 4:    # we have more than one channel
    color_mode = 'rgb'
    number_of_channels = imarray.shape[1]
    padPSF = pad(normPSF, imarray[:,0,:,:])
        
    for index in range(number_of_channels):    # do the deconvolution for each channel
        
        plt.figure()
        plt.imshow(imarray[IMG_PLANE,index,:,:], cmap="gray")
    
        result = deconvolution_algorithm(processing_unit, imarray[:,index,:,:], padPSF, ITERATIONS, imarray[:,index,:,:])    # deconvolution
        
        # hyperstack creation
        result = np.reshape(result, (result.shape[0], 1, result.shape[1], result.shape[2]))       
        if index == 0:
            deconvoluted_image = result
        else:
            deconvoluted_image = np.concatenate((deconvoluted_image, result), axis = 1)
            
        plt.figure()
        plt.imshow(result[IMG_PLANE,0,:,:], cmap="gray")
                
else:
    color_mode = 'gray'
    number_of_channels = 1
    padPSF = pad(normPSF, imarray)
    
    deconvoluted_image = result = deconvolution_algorithm(processing_unit, imarray, padPSF, ITERATIONS, imarray)    # deconvolution

    if number_of_dimensions == 2:
        plt.figure()
        plt.imshow(imarray, cmap="gray")
        plt.figure()
        plt.imshow(deconvoluted_image, cmap="gray")
    else:    # number_of_dimensions == 3
        plt.figure()
        plt.imshow(imarray[IMG_PLANE,:,:], cmap="gray")
        plt.figure()
        plt.imshow(deconvoluted_image[IMG_PLANE,:,:], cmap="gray")
        
deconvoluted_image = np.array(deconvoluted_image, dtype=np.float32)

tif.imsave(full_name + '_deconvoluted_iter' + str(ITERATIONS) + '.tif', deconvoluted_image, imagej = True, resolution = (10, 10), metadata = {'spacing': 0.2, 'unit': 'um', 'mode': color_mode, 'channels': number_of_channels})
