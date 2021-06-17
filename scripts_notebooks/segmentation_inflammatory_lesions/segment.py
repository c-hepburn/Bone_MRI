import numpy as np
import nibabel as nib
from skimage import measure

class Segment():
    '''
    Purpose:  peform image intensity-based segmentation via two thresholds within a region of
              interest.
              
    Input: 
    - (STIR) image, nifti file;
    - mask_1, nifti file; (normal bone segmentation); thresholds are estimated from the intensity
      distribution of the normal bone voxels;
    - mask_2, nifti file; (disease region segmentation); intensity-based segmentation is performed
      within the disease region;
    
    N.B.: pay attention to the order, in which the inputs are passed to the class;
    
    Output: nifti file(segmentation of abnormal bone voxels); the output array elements can take the
    following values:
        - 0 (background);
        - 1 (voxel was segmented at the lower threshold, i.e.,
            lower threshold <= voxel intensity < upper threshold);
        - 2 (voxel was segmented at the lower and upper thresholds, i.e.,
          lower threshold < upper threshold <= voxel intensity);
    
    Additional information can be retrieved from the class attributes, to which information is 
    stored after the execution of the code:
    - header: nifti header from the mask_1 nifti file;
    - datatype: data type from the mask_1 nifti file;
      header and data type are used to store the output in nifti file format;
    - intensities: 1D numpy array; contains STIR intensities of voxels considered to be normal 
      bone; following is determined from the normal bone voxels intensity distribution:
    - th_1: value of the lower threshold (float);
    - th_2: value of the upper threshold (float);
    - upper_q: upper quartile (float);
    - IQR: interquartile range (float);
    - multiple: multiple of the interquartile range;
      Upper quartile, IQR and multiple are used to compute the lower threshold defined as 
      upper_q + multiple * IQR;
    - seg_1: segmentation, produced at the lower threshold (nifti file); contains binary 3D numpy
             array of size [Height,Width,Depth];
    - seg_2: segmentation, produced at the upper threshold (nifti file); contains binary 3D numpy
             array of size [H,W,D];
    - seg_sum: sum of the two segmentations (nifti file); contains 3D numpy array of size [H,W,D];
      possible values of the array elements: 0,1,2;
    '''
    def __init__(self):
        'Initialize class attributes;'
        self.header = None
        self.datatype = None
        self.intensities = None
        self.th_1 = None
        self.th_2 = None
        self.upper_q = None
        self.IQR = None
        self.multiple = None
        self.seg_1 = None
        self.seg_2 = None
        self.seg_sum = None
        
    def extract_arrays(self, nifti_image, nifti_mask_1, nifti_mask_2):
        '''
        Purpose:
            - extract 3D numpy arrays from the provided nifti files; 
            - extract header and data type from the mask_1 nifti file; store header and data type to
              the corresponding class attributes;
        
        Input: 
            - (STIR) image, nifti file;
            - mask_1, nifti file; (normal bone segmentation);
            - mask_2, nifti file; (disease region segmentation);
      
        Output:
            - image, 3D numpy array of size [H,W,D];
            - mask_1, binary 3D numpy array of size [H,W,D];
            - mask_2, binary 3D numpy array of size [H,W,D];
            - header and datatype from mask_1 nifti file are stored in the corrsponding class
              attributes;
        '''
        ## Copy nifti header and datatype from the mask_1 nifti file;
        header = nifti_mask_1.header
        datatype = nifti_mask_1.get_data_dtype()
        
        ## Store the header and datatype to the class attributes;
        self.header = header
        self.datatype = datatype
        
        ## Retrieve numpy arrays from the nifti files;
        image = nifti_image.get_fdata()
        mask_1 = nifti_mask_1.get_fdata()
        mask_2 = nifti_mask_2.get_fdata()
        
        return image, mask_1, mask_2
    
    def retrieve_intensity_values(self, image, mask):
        '''
        Purpose: store intensities of voxels at specific location in the image to a 1D numpy array;

        Input:
            - (STIR) image, 3D numpy array of size [H,W,D];
            - mask_1, binary 3D numpy array of size [H,W,D]; (normal bone segmentation);

        Ouput: 1D numpy array;
        '''
        ## Find indices (coordinates) of voxels with label 1 in the mask;
        ## Output: tuple with three 1D numpy arrays, (J,I,K);
        index_arrays = np.where(mask == 1.)

        ## Create empty list to store intensities, retrieved from the image;
        intensities = []

        #### ----------------------- Loop over elements of the index arrays -----------------------
        ## (Determine the number of elements in the first index array);
        for n in range(index_arrays[0].shape[0]):

            ## Retrieve indices (coordinates);
            j,i,k = index_arrays[0][n], index_arrays[1][n], index_arrays[2][n]

            ## Retrieve intensity value of voxel at the current location from the image;
            intensity = image[j,i,k]

            ## Store intentisy value to the list;
            intensities.append(intensity)
        #### ----------------------------------------- Loop over elements of the index arrays ends;  

        ## Convert the list to a numpy array to be able to compute statistics;
        intensities = np.asarray(intensities)
        
        return intensities

    def compute_thresholds(self, array):
        '''
        Purpose: compute thresholds from the given intensity distribution;
        
        The lower threshold is defined as upper quartile + multiple * interquartile range;
        Multiple is computed by incrementing the default value 1.5 such that the difference in
        thresholds is less than half of the interquartile range (IQR), i.e.,
        multiple := multiple + 0.05;
        If the difference is less than IQR/2 (>0) for the default value, the computed threshold is
        returned directly;
        If difference <= 0, the lower threshold is re-computed by decreasing the multiple, i.e.,
        multiple =: multiple - 0.05 until the difference is <= IQR/2;
        (this situation is less likely to occur as the data showed skewed distributions);
        
        The upper threshold is the maximum value of the array;
        
        Input: 1D numpy array; (STIR intensities of the normal bone voxels);

        Output:
        - threshold 1, float;
        - threshold 2, float;
        - upper quartile, float;
        - multiple of IQR, float;
        - IQR, float;
        '''
        
        ## -------------------- Compute the lower threshold;
        
        ## Compute the lower and upper quartiles, i.e., 25th and 75th percentiles;
        ## Note: default interpolation scheme: linear;
        lower_q, upper_q = np.percentile( array, np.asarray( [25., 75.] ) )

        ## Compute the interquartile range (the difference between the upper and lower quartiles);
        IQR = upper_q - lower_q

        ## Compute the lower threshold the default (empirical) value of the multiple;
        multiple = 1.5
        th_1 = upper_q + multiple * IQR

        ## -------------------- Compute the upper threshold;
        th_2 = array.max()
        
        ## -------------------- Compute the difference in the thresholds;
        diff = th_2 - th_1

        ## Condition on the difference: if 0 < IQR/2 <= th_2 - th_1, 
        ## re-compute the lower threshold by incrementing the multiple;
        while diff >= IQR / 2.:

            ## Intrement the multiple;
            multiple += 0.05

            ## Compute the lower threshold;
            th_1 = upper_q + multiple * IQR

            ## Compute the difference in thresholds;
            diff = th_2 - th_1
        
        ## Condition on the difference: if difference <= 0, 
        ## re-compute the lower threshold by decreasing the multiple;
        while diff <= 0.:
            
            ## Decrease the multiple;
            multiple -= 0.05
            
            ## Compute the lower threshold;
            th_1 = upper_q + multiple * IQR

            ## Compute the difference in thresholds;
            ## Note: round the output, otherwise loop will end when the computed 
            ## difference is very close to zero, for example, 1e-15;
            diff = np.round(th_2 - th_1, 3)

        return th_1, th_2, upper_q, multiple, IQR
        
    def threshold(self, image, mask, th_1, th_2):
        '''
        Purpose: perform intensity-based segmentation at two given thresholds within the region of
                 interest;

        Input:
        - (STIR) image, 3D numpy array of size [H,W,D];
        - mask, binary 3D numpy array of size [H,W,D]; (disease region segmentation);
        - lower threshold, float;
        - upper threshold, float;

        Output:
        - binary 3D numpy array of size [H,W,D]; (segmentation at the lower threshold);
        - binary 3D numpy array of size [H,W,D]; (segmentation at the upper threshold);
        - 3D numpy array of size [H,W,D]; (sum of segmentations); possible voxel values: 0,1,2;
        '''
        ## Create numpy arrays filled with zeros of the input image size;
        seg_1 = np.zeros(image.shape)
        seg_2 = np.zeros(image.shape)

        ## Find indices (coordinates) of voxels with label 1 in the mask;
        ## Output: tuple with three 1D numpy arrays, (J,I,K);
        index_arrays = np.where(mask == 1.)

        #### ---------------------- Loop over elements of the index arrays ----------------------
        ## (Determine the number of elements in the first index array);
        for n in range(index_arrays[0].shape[0]):

            ## Retrieve indices (coordinates);
            j,i,k = index_arrays[0][n], index_arrays[1][n], index_arrays[2][n]

            ## Retrieve intensity value of voxel at the current location from the image;
            intensity = image[j,i,k]

            ## Condition on the intensity: if the value is above or equal to the lower threshold,
            ## assign label 1 to the voxel at the current location in the corresponding array;
            if intensity >= th_1:
                seg_1[j,i,k] = 1.

            ## Condition on the intensity: if the value is above or equal to the upper threshold,
            ## assign label 1 to the voxel at the current location in the corresponding array;
            if intensity >= th_2:
                seg_2[j,i,k] = 1.
        #### --------------------------------------- Loop over elements of the index arrays ends; 

        ## Sum the generated segmentations;
        ## Output: 3D numpy array; possible voxel values: 0,1,2;
        seg_sum = seg_1 + seg_2

        return seg_1, seg_2, seg_sum
    
    def remove_residues(self, seg_1, seg_sum):
        '''
        Purpose: remove regions consisting of less than five connected pixels (possible pixel values
                 1. or 2.) from the given array;

        Input:
        - binary 3D numpy array of size [H,W,D]; (segmentation at the lower threshold);
        - 3D numpy array of size [H,W,D]; possible voxel values: 0,1,2; (sum of segmentations at the
          lower and upper thresholds);

        Output: 3D numpy array of size [H,W,D]; (sum of segmentations at the lower and upper
                thresholds with removed on each slice regions consisting of <5 connected pixels);
        '''

        ## Create numpy array filled with zeros of the input segmentation size;
        seg_sum_res_removed = np.zeros(seg_1.shape)

        ## Loop over slices;
        ## - group the connected pixels into individual regions by assigning the same unique label;
        ## - determine the size of the individual regions;
        ## - disregard regions that consist of less or four pixels;
        
        ## -------------------------------------------------------------------- Loop over slices;
        ## (Determine number of slices in the segmentation);
        for k in range(seg_1.shape[-1]):

            ## Retrieve segmentation slice;
            seg_1_slice = seg_1[:,:,k]

            ## Assign a unique label to the connected pixels; note: segmentation at the lower
            ## threshold is a union of the segmentations at the lower and upper thresholds;
            ## Input: binary 2D numpy array of size [H,W]; Output: 
            ## - 2D numpy array, possible pixel values are in the range (0, number of labels);
            ## - number of labels, which equals the maximum label value, i.e., label 0 (background)
            ## is not taken into account);
            labelled_array, labels_num = measure.label(seg_1_slice,
                                                       connectivity = 2,
                                                       return_num = True)
            
            ## --------------------------------------------------------------- Loop over labels; 
            ## Note: python counts from 0 and excludes the last number in the range,
            ## therefore 1 is added, otherwise the last label is missed;
            for label in range(labels_num + 1):

                ## Exclude label 0 (background);
                if label == 0:
                    pass

                else:
                    ## Find indices (coordinates) of pixels with the current label in the labelled
                    ## array; output: tuple with 2 index numpy arrays, (J,I);
                    label_coord = np.where(labelled_array == label)
                    
                    ## Determine number of elements in the first index array,
                    ## this gives number of elements in the region of connected pixels;
                    pixnum = label_coord[0].shape[0]
                
                    ## Condition on the size of the current region: if number of pixels is 
                    ## more or equal to 4, 'store' the region in the corresponding array;
                    if  pixnum >= 4:

                        ## ----------- Loop over the coordinates of pixels with the current label;
                        for n in range(pixnum):

                            ## Retrieve indices (coordinates);
                            j,i = label_coord[0][n], label_coord[1][n]

                            ## Assign value of the pixel at the current location in the seg_sum
                            ## to the pixel at the corresponding location in the array;
                            seg_sum_res_removed[j,i,k] = seg_sum[j,i,k]
                        ## ---------------------------------------------------------------------
            ## ---------------------------------------------------------- Loop over labels ends;
        ## -------------------------------------------------------------- Loop over slices ends;
        return seg_sum_res_removed
    
    def segment(self, nifti_image, nifti_mask_1, nifti_mask_2):
        '''
        Purpose: perform intensity-based segmentation at two given thresholds within the region of
                 interest;
        
        Input: 
            - (STIR) image, nifti file;
            - mask_1, nifti file; (normal bone segmentation);
            - mask_2, nifti file; (disease region segmentation);
        
        Output:
            - nifti file(segmentation of abnormal bone voxels); the output array elements can take
              values 0,1,2;
            - intermediate results are stored in the corresponding class attributes;
        '''
        ## Retrieve 3D numpy arrays from nifti files;
        image, mask_1, mask_2 = self.extract_arrays(nifti_image, nifti_mask_1, nifti_mask_2)
        
        ## Retrieve intensity values of voxels considered to be normal bone;
        intensities = self.retrieve_intensity_values(image, mask_1)
        
        ## Compute thresholds, given normal bone intensity distribution;
        th_1, th_2, upper_q, multiple, IQR = self.compute_thresholds(intensities)
        
        ## Segment within the disease region, using the computed thresholds;
        seg_1, seg_2, seg_sum = self.threshold(image, mask_2, th_1, th_2)
        
        ## Remove residues from the sum of generated segmentations;
        seg_sum_res_removed = self.remove_residues(seg_1, seg_sum)
        
        ## Change data type of the arrays to store data in nifti file format;
        ## (when loaded with nibabel, mask data type (unit16) is converted to float64);
        seg_1= seg_1.astype(self.datatype)
        seg_2 = seg_2.astype(self.datatype)
        seg_sum = seg_sum.astype(self.datatype)
        seg_sum_res_removed = seg_sum_res_removed.astype(self.datatype)
        
        ## Create nifti files using header from the provided file;
        seg_1 = nib.Nifti1Image(seg_1, None, header = self.header)
        seg_2 = nib.Nifti1Image(seg_2, None, header = self.header)
        seg_sum = nib.Nifti1Image(seg_sum, None, header = self.header)
        seg_sum_res_removed = nib.Nifti1Image(seg_sum_res_removed, None, header = self.header)
        
        ## Store the intermediate results to the class attributes;
        self.intensities = intensities
        self.th_1 = th_1
        self.th_2 = th_2
        self.upper_q = upper_q
        self.multiple = multiple
        self.IQR = IQR
        self.seg_1 = seg_1
        self.seg_2 = seg_2
        self.seg_sum = seg_sum
        
        return seg_sum_res_removed