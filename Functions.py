import numpy as np
import cupy as cp
from numpy.fft import ifftshift, fftn, ifftn
import time



# DECONVOLUTION ALGORITHM
def deconvolution_algorithm (processing_unit, image, psf, iterations, guess):
    
    '''
    This function calls the correct RL deconvolution function according to the
    processing unit selected and print the working time
    
    Parameters:
        processing_unit: can be CPU or GPU
        image: n-dimensional array which represent the image to deconvolve
        psf: n-dimensional array which represent the psf function
        iterations: number of iterations
        guess: initial guess of the algorithm. It can be an array with
               dimensions equal to those of the image or a number, int or float,
               which will be the value of a matrix with image dimensions with
               all equal elements. Hint: use as initial guess the image to
               deconvolve itself or a matrix of all 0.5 (thus pass the
               parameter 0.5 to the function)
        
    Return:
        result: deconvoluted image
    '''
    if processing_unit == "GPU":
        start = time.time()
        result = cupyRL(image, psf, iterations, guess)    # deconvolution
        end = time.time()
        print("Time required with cupy: ", (end-start))
    
    elif processing_unit == "CPU":
        start = time.time()
        result = RichardsonLucy(image, psf, iterations, guess)    # deconvolution
        end = time.time()
        print("Time required with numpy: ", (end-start))
    
    else:
        raise NameError("Select GPU or CPU")
    
    #tif.imsave(file_name + '_channel_' + str(index) + '_deconvoluted.tif', result, resolution=(10, 10))
    
    return result



# RICHARDSON-LUCY DECONVOLUTION FUNCTION WITH CPU
def RichardsonLucy (image, psf, iterations, guess):
    
    '''
    This function do the Richardson-Lucy deconvolution of an image
    
    Parameters:
        image: n-dimensional array which represent the image to deconvolve
        psf: n-dimensional array which represent the psf function
        iterations: number of iterations
        guess: initial guess of the algorithm. It can be an array with
               dimensions equal to those of the image or a number, int or float,
               which will be the value of a matrix with image dimensions with
               all equal elements. Hint: use as initial guess the image to
               deconvolve itself or a matrix of all 0.5 (thus pass the
               parameter 0.5 to the function)
        #reg_lambda: if it is a number, the Total Variation regularization is
                    #implemented and this value will be the regularization
                    #parameter (lambda). Otherwise, if reg_lambda == 'noTV', the
                    #TV regularization will not be implemented

    Return:
        deconv: deconvoluted image
    '''
    reg_lambda = 2.5    # regularization parameter for the TV regularization
    identity = np.ones(image.shape)
    norm = []    # dataset of Frobenius norm at each iteration
    #plot_position = 1    # position of the plot in the already opened figure (in the main)
                         # PUT 1 IF WE DON'T PLOT THE CONVOLUTED IMAGE OR WE HAVE ONE LESS IMAGE, PUT 2 OTHERWISE
    
    # if the last parameter is a number, create a matrix with all equal values as initial guess
    if type(guess) == int or type(guess) == float:
        deconv = np.full(image.shape, guess)
    else:
        deconv=guess
        
    psf_trans = np.flip(psf)    # transpose matrix of psf array
    
    #RL algorithm
    for i in range (iterations):
        TVdeconv = deconv    # comment this line if you don't implement TV regularization
        
        convol = conv(deconv, psf)
        residual = image / convol
        error = conv(residual, psf_trans)
        deconv = deconv * error
        # until here simply RL deconvolution
        '''
        # TV regularization (COMMENT THIS PART IF YOU WANT ONLY PURE RL ALGORITHM)
        #if type(reg_lambda) == int or type(reg_lambda) == float:
        zgrad, ygrad, xgrad = np.gradient(TVdeconv)
        znorm = np.linalg.norm(zgrad)
        ynorm = np.linalg.norm(ygrad)
        xnorm = np.linalg.norm(xgrad)
        z = zgrad / znorm
        y = ygrad / ynorm
        x = xgrad / xnorm
        zdiv = np.gradient(z, axis=0)
        ydiv = np.gradient(y, axis=1)
        xdiv = np.gradient(x, axis=2)
        TV = reg_lambda * (zdiv + ydiv + xdiv)    # multiplication factor to introduce TV regularization
        deconv = deconv * 1/(identity - TV)
        '''
        #norm.append(np.linalg.norm(convol - image))    # Frobenius norm of the residual blur
    '''    
    # plot of the Frobenius norm at each iteration   
    # maybe I can also plot a figure to show the minimum...like a U curve...     
    plt.figure()
    plt.plot(range(iterations), norm)
    plt.title('Frobenius norm')
    plt.show()
    #print("Min Frobenius norm", min(norm))
    '''
    return deconv




# RICHARDSON-LUCY DECONVOLUTION FUNCTION WITH GPU
def cupyRL (image, psf, iterations, guess):
    '''
    Same as RichardsonLucy function but done in cupy
    '''
    norm = []    # dataset of Frobenius norm at each iteration
    #plot_position = 1    # position of the plot in the already opened figure (in the main)
                         # PUT 1 IF WE DON'T PLOT THE CONVOLUTED IMAGE OR WE HAVE ONE LESS IMAGE, PUT 2 OTHERWISE
    
    # if the last parameter is a number, create a matrix with all equal values as initial guess
    if type(guess) == int or type(guess) == float:
        deconv = cp.full(image.shape, guess)
    else:
        deconv=guess
        
    psf_trans = np.flip(psf)    # transpose matrix of psf array
    
    image = cp.asarray(image)
    psf = cp.asarray(psf)
    psf_trans = cp.asarray(psf_trans)
    deconv = cp.asarray(deconv)

    
    #RL algorithm
    for i in range (iterations):
        convol = cupyconv(deconv, psf)
        residual = image / convol
        error = cupyconv(residual, psf_trans)
        deconv = deconv * error
        
        #norm.append(cp.linalg.norm(convol - image))    # Frobenius norm of the residual blur
    '''
    # plot of the Frobenius norm at each iteration   
    # maybe I can also plot a figure to show the minimum...like a U curve...  
    norm_cpu = cp.asnumpy(norm)
    plt.figure()
    plt.plot(range(iterations), norm_cpu)
    plt.title('Frobenius norm')
    plt.show()
    #print("Min Frobenius norm", min(norm_cpu))
    '''
    deconv = cp.asnumpy(deconv)
    return deconv





# BACKGROUND REMOVAL FUNCTION
'''
This function take as input a n-dimensional array and set to zero all the
elements smaller than a fixed threshold

Parameters:
    array: n-dimensional array
    threshold: value under which an element is put to zero

Return:
    back: background-free array
'''
def backfree (array, threshold):
    back = (array > threshold) * array
    # the parenthesis return 1 or 0 if array > threshold or viceversa, respectively
    return back



# NORMALIZATION FUNCTION
'''
This function take as input an n-dimensional array and normalize it to 1

Parameter:
    array: n-dimensional array to normalize
Return:
    normal: normalized array
'''
def norm (array):
    normal = array / np.sum(array)
    return normal



# ZERO-PADDING FUNCTION
'''
This function takes as input two n-dimensional arrays, one with smaller
dimesnions wrt the other. The aim is to bring the smaller array to the same
dimensions of the bigger one adding zeros. The result will be a bigger array,
with the desired dimensions of reference, where the original array is placed
in the middle of the zero-padded array (zeros are placed all around the
original array to pad)

Parameters:
    array: n-dimensional array to pad
    reference: n-dimensional array with the desired dimensions in which put array
Return:
    result: padded array
'''
def pad (array, reference):
    if array.shape == reference.shape:    # if same dimensions, do nothing
        return array
    
    result = np.zeros(reference.shape, dtype=np.float32)
    offset = np.zeros(reference.ndim, dtype=np.uint8)
    
    # definition of the offsets -> position of the void matrix (result) in which put our array
    for i in range (reference.ndim):
        offset[i] = int((reference.shape[i] - array.shape[i]) / 2)
        
    # create a list of slices from offset to offset + shape in each dimension
    insert = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range (reference.ndim)]
    
    # insert array in result at the specified position given by offsets
    result[insert] = array
    return result



# ZERO-PADDING FUNCTION USING NUMPY.PAD (slower than the other function...)
'''
This function takes as input two 2D/3D arrays, one with smaller
dimesnions wrt the other. The aim is to bring the smaller array to the same
dimensions of the bigger one adding zeros. The result will be a bigger array,
with the desired dimensions of reference, where the original array is placed
in the middle of the zero-padded array (zeros are placed all around the
original array to pad)

Parameters:
    array: 2 or 3-dimensional array to pad
    reference: 2 or 3-dimensional array with the desired dimensions
Return:
    result: padded array
'''
def padv2 (array, reference):
    if array.shape == reference.shape:    # if same dimensions, do nothing
        return array
    
    z = reference.shape[0] - array.shape[0]
    y = reference.shape[1] - array.shape[1]
    
    '''
    If dimensions along axis have the same parity we will have a perfectly
    centered result, otherwise we will have one more element on one side.
    
    The following piece of code creates the correct tuples to pass to np.pad
    function, according to the parity of dimensions, as explained before
    '''
    if z%2 == 0:
        zoff = (int(z/2), int(z/2))
    else:
        zoff = (int(z/2)+1, int(z/2))
        
    if y%2 == 0:
        yoff = (int(y/2), int(y/2))
    else:
        yoff = (int(y/2)+1, int(y/2))
        
    if array.ndim == reference.ndim == 3:    # add the third coordinate if we have a 3D array
        x = reference.shape[2] - array.shape[2]
        
        if x%2 == 0:
            xoff = (int(x/2), int(x/2))
        else:
            xoff=(int(x/2)+1, int(x/2))
            
        result = np.pad(array, (zoff, yoff, xoff), 'constant')    # result for 3D arrays
        return result
        
    result = np.pad(array, (zoff, yoff), 'constant')    # result for 2D arrays
    return result



# FFT CONVOLUTION
'''
This function convolve two n-dimensional arrays using FFT. The two inputs must
have the same dimensions/shapes

Parameters:
    array1: first input (can be an image)
    array2: second input (can be a PSF)
Return:
    result: convoluted array
    ft1: FFT of the array1
    ft2: FFT of the array2
'''
def conv(array1, array2):
    # first step: do the Fourier transform of the two arrays
    ft1 = fftn(array1)
    ft2 = fftn(array2)

    # second step: do the product in the Fourier domain and the inverse transform
    product = ifftshift(ifftn(ft1 * ft2))
    result = np.abs(product)    # coming back to the image
    return result #, ft1, ft2



# FFT CONVOLUTION
'''
Same of conv function but done with cupy
'''
def cupyconv(array1,array2):
    ft1 = cp.fft.fftn(array1, s=None, axes=None, norm=None)
    ft2 = cp.fft.fftn(array2, s=None, axes=None, norm=None)
    product = ft1*ft2
    inverse = cp.fft.ifftshift((cp.fft.ifftn(product, s=None, axes=None, norm=None)), axes=None)
    result = cp.abs(inverse)
    return result