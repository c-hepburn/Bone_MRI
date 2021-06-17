import numpy as np
import scipy.linalg as la
import scipy.ndimage as nd
import elasticdeform
import torch.nn.functional as F
import torch

class Transform():
    '''
    Purpose: perform the following transformations on images and corresponding masks
        - affine (rotation, shearing, scaling);
        - elastic deformation;
        - random flip with probability 0.5;
        - gamma correction (on images only);
             
    Inputs: 
        - image batch, 4D torch tensor [N,1,H,W]; normalized data; 
         (N - batch size, i.e., number of input image slices, 1 - default number of input channels,
          H - input plane height and W - input plane width in pixels);
        - mask batch, 4D torch tensor [N,1,H,W];
    
    Outputs:
        - transformed image batch, 4D torch tensor [N,1,H,W];
        - transformed mask batch,  4D torch tensor [N,1,H,W];
    
    Note: tensor device is converted to 'cpu', as transformations are applied to numpy arrays 
          only. If device_mode = 'cuda', the output will be allocated on GPU; 
    
    Class attributes:
        - dim: input batch dimensions;
        - order: spline interpolation order, default = 0;
        - warp_mode: points outside the boundaries of the input are filled according to the given
          mode, default = 'mirror', otherwise: ‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’;
        - device_mode: if 'cuda' the output is transferred to GPU;
        - ranges: dictionary, default: None;
                Example of ranges to pass to the Transform class:
                ranges = {}
                ranges['degree'] = 5 => range: (-5,5);
                ranges['scale_yx'] = (0.9,1.2) => range: (0.9,1.2);
                ranges['shear_yx'] = 0.2 => range (-0.2,0.2);
                ranges['sigma_points'] = (3,14) for elastic deformation field
        
        - parameters: dictionary, that stores the actual values of the transformation parameters.
          Values are stored only after calling the transform() or batch_transform() methods;
    '''
    def __init__(self,
                 input_dim = None,
                 order = 0,
                 warp_mode = 'mirror',
                 device_mode = 'cuda',
                 ranges = None):
        
        ## Initialize class attributes;
        self.dim = input_dim
        self.order = order
        self.warp_mode = warp_mode
        self.device_mode = device_mode
        self.ranges = ranges
        self.parameters = {}

    def rotation(self, deg):
        '''
        Input: angle in degrees;
        Output: rotation matrix, numpy 2D array of size [2,2], data type: float64;
        Note: rotation matrix takes into account a non-standard 2D coordinate system (y axis
              increases downwards), i.e., positive angle implies clockwise rotation;
        '''
        ## Convert degree to radians as np. trigonometric functions require input angle in radians;
        rad = np.radians(deg)
        
        return np.array([
            [np.cos(rad), np.sin(rad)],
            [-np.sin(rad), np.cos(rad)]
        ])

    def scaling(self, sy, sx):
        '''
        Input:
            - sy, scaling factor along y axis;
            - sx, scaling factor along x axis;
        
        (apply identity transformation if no scaling is intended, i.e., sy = sx = 1);

        Output: scaling matrix, 2D numpy array of size [2,2];
        '''
        return np.array([
            [sy,0],
            [0,sx]
        ])

    def shear(self, shy, shx):
        '''
        Input:
        - shear factor along y axis;
        - shear factor along x axis;
        
        Output: shear matrix, 2D numpy array of size [2,2];
        '''
        return np.array([
            [1,shy],
            [shx,1]
        ])

    def compose(self):
        '''
        Purpose: compose affine transformation matrix from rotation, scaling and shear matrices;
        Outputs: affine transformation matrix, 2D numpy array of size [2,2];
        Note: affine transformation parameters are randomly generated from uniform distribution of
              the given ranges;
        '''
        ## Retrieve ranges for affine transformation parameters;
        deg = self.ranges['degree']
        scale_min, scale_max = self.ranges['scale_yx']
        shear = self.ranges['shear_yx']
        
        ## Randomly sample parameters from a uniform distribution of the given ranges;
        deg = np.random.uniform(low = -deg, high = deg)
        sy = np.random.uniform(low = scale_min, high = scale_max)
        sx = np.random.uniform(low = scale_min, high = scale_max)
        shy = np.random.uniform(low = -shear, high = shear)   
        shx = np.random.uniform(low = -shear, high = shear)
            
        ## Store the parameters in the dictionary;
        self.parameters['degree'] = deg
        self.parameters['scale_yx'] = (sy,sx)
        self.parameters['shear_yx'] = (shy,shx)
        self.parameters['sigma_points'] = self.ranges['sigma_points']
        
        ## Return composed affine transformation matrix;
        return np.dot(self.rotation(deg), np.dot(self.shear(shy,shx), self.scaling(sy,sx)) )

    def offset(self, T):
        '''
        Purpose: compute offset coordinates vector;
        
        Affine transformations are applied with respect to the image center, i.e., image origin 
        is shifted to center of the image. Given the target image coordinate vector v,
        the pixel value is determined from the input image at the position
        np.dot(inverse coord. transformation matrix, v) + offset;
        
        Input: transformation, 2D numpy array of size [2,2];
        Output: offset coordinates vector, 1D numpy array;
        '''
        ## Determine image center coordinates. Output: 2D numpy array of size [2,1];
        p = np.array([ [self.dim[-1]//2], [self.dim[-1]//2] ])
        
        ## Compute offset;
        ## Apply inverse transform on target image center coordinates (same as source image)
        ## The resulting offset vector is the difference between source image center coordinates
        ## and transformed coordinates of the target image center; 
        offset = p - np.dot(la.inv(T),p)
        
        ## Output: 1D array of size [2];
        return offset.flatten() 
        
    def affine(self, image_slice, mask_slice):
        '''
        Purpose: apply the affine transformation to the input arrays;
        
        Inputs:
            - image slice, 2D numpy array of size [H,W];
            - mask slice (binary image), 2D numpy array of size [H,W];
            
        Outputs:
            - transformed image slice,  2D numpy array of size [H,W];
            - transformed mask slice (binary image), 2D numpy array of size [H,W];
        
        Note: nd.affine_transform does "pull" resampling, transforming the output space 
        to the input space to locate data. Therefore, inverse of transformation matrix is used;
        '''
        ## Define an affine transformation with random parameters;
        ## Offset: transformation is performed with respect to the image center;
        T = self.compose()  
        
        ## Apply the affine transformation to the image and mask slices
        def_image = nd.affine_transform(image_slice, la.inv(T), 
                                        order = self.order, 
                                        mode = self.warp_mode,
                                        offset = self.offset(T))
        
        def_mask = nd.affine_transform(mask_slice, la.inv(T),
                                       order = self.order, 
                                       mode = self.warp_mode,
                                       offset = self.offset(T)) 
        
        return def_image, def_mask
    
    def elastic(self, image_slice, mask_slice):
        '''
        Purpose: apply elastic deformation to the input arrays;
        
        Inputs:
            - image slice, 2D numpy array of size [H,W];
            - mask slice (binary image), 2D numpy array of size [H,W];
            
        Outputs:
            - transformed image slice,  2D numpy array of size [H,W];
            - transformed mask slice (binary image), 2D numpy array of size [H,W];
        '''
        ## Retrieve deformation parameters
        sigma, points = self.ranges['sigma_points']
        
        ## Apply elastic deformation on images
        def_image, def_mask = elasticdeform.deform_random_grid([image_slice, mask_slice],
                                                         sigma = sigma,
                                                         points = points,
                                                         order = self.order, 
                                                         mode = self.warp_mode)
        return def_image, def_mask
    
    
    def flip(self, image_slice, mask_slice):
        '''
        Purpose: reverse the order of the elements in the input arrays along the specific axis
        (horizontal flip);
        
        Inputs:
            - image slice, 2D numpy array of size [H,W];
            - mask slice (binary image), 2D numpy array of size [H,W];
            
        Outputs:
            - transformed image slice,  2D numpy array of size [H,W];
            - transformed mask slice (binary image), 2D numpy array of size [H,W];
        '''
        return np.flip(image_slice, axis = 1), np.flip(mask_slice, axis = 1)
    
    def gamma_correction(self, image_slice):
        '''
        Purpose: raise the input array elements to a random power (gamma correction);
        Input: image slice, 2D numpy array of size [H,W];  
        Output: transformed image slice, 2D numpy array of size [H,W];
        '''
        ## Generate a random number from uniform distribution of the pre-defined ranges
        power = np.random.uniform(low = 0.6, high = 1.2)
        
        ## Store power value to the dictionary;
        self.parameters['power'] = power
        
        ## Raise the input array elements to the power
        return image_slice**power
        
    def transform(self, image_slice, mask_slice):
        '''
        Purpose: apply the following transformations to the input arrays (image and mask slices)
            - elastic deformation;
            - affine transformation;
            - horizontal flip with probability 0.5;
            - gamma correction (image slice only);
            
        Inputs:
            - image slice, 2D numpy array of size [H,W];
            - mask slice (binary image), 2D numpy array of size [H,W];
            
        Outputs:
            - transformed image slice,  2D numpy array of size [H,W];
            - transformed mask slice (binary image), 2D numpy array of size [H,W];
        '''
        ## Apply elastic deformation;
        def_image_slice, def_mask_slice = self.elastic(image_slice, mask_slice)
        
        ## Apply affine transformation;
        def_image_slice, def_mask_slice = self.affine(def_image_slice, def_mask_slice)
        
        ## Sample a number from a uniform distribution within the range (0,1);
        p = np.random.rand()
        
        ## If greater than 0.5, flip the arrays horizontally;
        if p > 0.5:
            def_image_slice, def_mask_slice = self.flip(def_image_slice, def_mask_slice)
        
        ## Apply gamma correction to the transformed image array;
        def_image_slice = self.gamma_correction(def_image_slice)
        
        return def_image_slice, def_mask_slice

    def batch_transform(self, image, mask):
        '''
        Purpose: apply the following transformations to the input image and mask batches
            - elastic deformation;
            - affine transformation;
            - horizontal flip with probability 0.5;
            - gamma correction (image only);
            
        Inputs:
            - image batch, 4D torch tensor of size [N,1,H,W]; normalized data;
            - mask batch, 4D torch tensor of size [N,1,H,W];
            
        Outputs:
            - transformed image batch, 4D torch tensor of size [N,1,H,W];
            - transformed mask batch, 4D torch tensor of size [N,1,H,W];
        '''
        ## Raise error if input (image or mask) batch dimensions are not known;
        assert(self.dim != None) 
        
        ## Allocate tensor on cpu to apply transformations;
        if not image.device == 'cpu': image = image.cpu()
        if not mask.device == 'cpu': mask = mask.cpu()
            
        ## Convert torch tensors to numpy arrays;
        image, mask = image.numpy(), mask.numpy()
        
        ## Create empty 4D numpy arrays of size [N,1,H,W] to store the transformed
        ## image and mask slices;
        def_image = np.zeros(image.shape, dtype = image.dtype)
        def_mask = np.zeros(mask.shape, dtype = mask.dtype )

        #### Loop over batch dimension to apply the transformations on each image, mask slice;
        for k in range(image.shape[0]):
            
            ## Retrieve image and mask slice;
            image_slice, mask_slice = image[k,0,:,:], mask[k,0,:,:]
            
            ## Apply transformations;
            def_image_slice, def_mask_slice = self.transform(image_slice, mask_slice)
            
            ## Store the transformed image and mask slices to the corresponding arrays;
            def_image[k,0,:,:], def_mask[k,0,:,:] = def_image_slice, def_mask_slice
        #### ---------------------------------------------------------------------------------
        
        ## Convert the numpy arrays to torch tensors;
        def_image, def_mask = torch.from_numpy(def_image), torch.from_numpy(def_mask)
        
        ## If specified, allocate tensors on GPU;
        if self.device_mode == 'cuda':
            def_image, def_mask = def_image.cuda(), def_mask.cuda()
            
        return def_image, def_mask