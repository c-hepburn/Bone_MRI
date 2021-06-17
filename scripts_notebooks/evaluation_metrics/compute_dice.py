import numpy as np

def compute_overlap(array_1, array_2):
    '''
    Purpose: compute Dice coefficient (overlap) for the foreground class (array elements with the
             value 1), given two binary images;
    
    Input: 
        - binary image, 2D or 3D numpy array of size [H,W] or [H,W,N];
        - binary image, 2D or 3D numpy array of size [H,W] or [H,W,N];
    
    Output: Dice coefficient,
        - float if at least one array contains foreground class elements;
        - None if both arrays contain background elements only;
    '''
    ## Compute number of the foreground elements in each array; 
    voxcount_1 = np.sum(array_1)
    voxcount_2 = np.sum(array_2)
    
    ## Compute sum of the foreground elements; 
    voxcount_sum =  voxcount_1 + voxcount_2
    
    ## Compute Dice;
    ## If both arrays contain background class elements only return None;
    if voxcount_sum == 0:
        
        overlap = None
        
    else:
        
        ## Compute number of the foreground elements that have the same location in both arrays;
        voxcount_intersection = np.sum( array_1 * array_2 )
    
        overlap = 2 * voxcount_intersection / voxcount_sum
    
    return overlap