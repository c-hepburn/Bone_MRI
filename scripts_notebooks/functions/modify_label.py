import numpy as np

def modify_label(array):
    '''
    Purpose: assign to the array elements with the value 2. the new value 1.;
    Input: 3D numpy array of size [Height,Width,Depth]; possible values for the array elements: 0,1,2;
    Output: binary 3D numpy array [H,W,D];
    '''
    ## Condition on the array elements with the value 2.: replace value with 1.;
    ## Condition on the array elements with the value 1.: do not change;
    new_array = np.where( (array == 2.) | (array == 1.), 1., 0. )
    
    return new_array