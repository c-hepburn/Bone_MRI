import numpy as np

def compute_vs(array_1, array_2):
    '''
    Purpose: volume similarity coefficient for the foreground class (array elements with the value
    1), given two binary images;
    
    Input: 
        - binary image, 3D numpy array of size [H,W,N];
        - binary image, 3D numpy array of size [H,W,N];
    
    Output:
    - volume similarity coefficient (float);
    - voxel count in the array 1 (integer);
    - voxel count in the array 2 (integer);
    '''
    ## Compute number of the foreground elements in each array; 
    voxcount_1 = np.sum(array_1)
    voxcount_2 = np.sum(array_2)
    
    ## Compute volume similarity coefficient;
    ## In case if both arrays do not contain the foreground elements raise error;
    assert (voxcount_1 + voxcount_2) != 0., 'Arrays do not contain the foreground class elements.' 
    
    vs = 1. - np.abs(voxcount_1 - voxcount_2) / (voxcount_1 + voxcount_2)
    
    return vs, voxcount_1, voxcount_2