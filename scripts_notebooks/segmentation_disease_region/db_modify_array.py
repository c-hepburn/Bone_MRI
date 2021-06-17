import numpy as np

def modify_array(array):
    '''
    Purpose: data processing
    - rotate 3D numpy array by 90d counterclockwise in the plane specified by axes;
    - add one dimension and modify dimensions order;
    - change data type to float32;
    
    Explanation: 
    - given data is stored in a specific way, counterclockwise rotation is for convenient
      visualization;
    - 2D convolution layers require 4D (torch) tensors as inputs; tensor size is given by
      [N,C,H,W],where N - batch size (i.e., number of image slices), C - number of input channels,
      H - input plane height and W - input plane width in pixels;
    - data type of input tensors must match data type of tensors, holding network parameters, i.e.,
      float32;
      
    Input: 3D numpy array, [H,W,N], data type float64 
    Output: 4D numpy array, [N,1,H,W], data type float32
    '''
    ## Rotate the array by 90 degrees counterclockwise
    array = np.rot90(array)
    
    ## Change array data type 
    array = array.astype('float32')

    ## Create new 4D numpy array of size [N,1,H,W], specify data type
    size = (array.shape[2], 1, array.shape[0], array.shape[1])
    new_array = np.zeros(size, dtype = 'float32')
    
    ## ----------------------------------------- Loop over array last dimension (slices)
    for k in range(array.shape[2]):
        
        ## Store current slice to the new array
        new_array[k,0,:,:] = array[:,:,k]
    ## ---------------------------------------------------------------------------------
    
    return new_array