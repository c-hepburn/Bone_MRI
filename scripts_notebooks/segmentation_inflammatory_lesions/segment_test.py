import unittest
from segment import *

## Create object of the Segment class;
segment = Segment()

class SegmentTest(unittest.TestCase):
    
    def setUp(self):
        'Purpose: generate inputs and the expected outputs to test the Segment class methods;'
        ## ------------------------------------------------------------  Generate input data;
        ## Generate image;
        image = np.zeros([6,6,1])
        image[0,:,0] = 10.
        image[2:,0,0] = 1.
        image[2:,1,0] = 5.
        image[2:,2,0] = 10.

        image[2,4,0] = 20.
        image[4:,4:,0] = 20.

        ## Generate mask;
        mask = np.zeros([6,6,1])
        mask[2:,:,0] = 1.
        
        ## Generate input for compute_thresholds() method;
        input_array = np.asarray([1,1,1,2,2,2,2,5,5,11])
        
        ## Generate inputs for threshold() method;
        th_1 = 5.
        th_2 = 10.
        
        ## ------------------------------------------------------- Generate the expected outputs;
        ## ---------------------- Expected output of the retrieve_intensity_values() method;
        array = np.asarray([0,0,0,0,0,0,0,
                            1,1,1,1,
                            5,5,5,5,
                            10,10,10,10,
                            20,20,20,20,20], dtype = float)
        
        ## ---------------------- Expected outputs of the threshold() method;
        ## Segmentation at lower threshold;
        seg_1 = np.zeros([6,6,1])
        seg_1[2:,1,0] = 1.
        seg_1[2:,2,0] = 1.
        seg_1[2,4,0] = 1.
        seg_1[4:,4:,0] = 1.
        
        ## Segmentation at upper threshold;
        seg_2 = np.zeros([6,6,1])
        seg_2[2:,2,0] = 1.
        seg_2[2,4,0] = 1.
        seg_2[4:,4:,0] = 1.
        
        ## Sum of the segmentations;
        seg_sum = seg_1 + seg_2
        
        ## ---------------------- Expected output of the remove_residues() method;
        seg = np.zeros([6,6,1])
        seg[2:,1,0] = 1.
        seg[2:,2,0] = 2.
        seg[4:,4:,0] = 2.

        self.image = image
        self.mask = mask
        self.input_array = input_array
        self.th_1 = th_1
        self.th_2= th_2
        self.array = array
        self.seg_1 = seg_1
        self.seg_2 = seg_2
        self.seg_sum = seg_sum
        self.seg = seg
        
    def test_retrieve_intensity_values(self):
        
        print('Generated input image\n', self.image[:,:,0])
        print('\nGenerated input mask\n', self.mask[:,:,0])
        print('\nExpected output of the retrieve_intensity_values() method\n', self.array)
        print('\nExpected outputs of the threshold() method (thresholds: 5., 10.)\n')
        print(self.seg_1[:,:,0], '\n')
        print(self.seg_2[:,:,0], '\n')
        print(self.seg_sum[:,:,0], '\n')
        print('\nExpected output of the remove_residues() method\n', self.seg[:,:,0])
        
        ## Call the function under test and pass generated data;
        result = segment.retrieve_intensity_values(self.image, self.mask)
        
        ## Sort elements of the array in ascending order to be able to compare
        ## with the expected result;
        sorted_result = np.sort(result)
        
        ## If the function output and the expected array are not equal, raise error;
        self.assertTrue( np.array_equal(self.array, sorted_result) )
        
    def test_compute_thresholds(self):
         
        ## Test case 1: initial th_1 > th_2;
        th_1, th_2, upper_q, multiple, IQR = segment.compute_thresholds(self.input_array)
        
        self.assertAlmostEqual(th_1, 9.65)
        self.assertAlmostEqual(th_2, 11.)
        self.assertAlmostEqual(upper_q, 4.25)
        self.assertAlmostEqual(multiple, 1.8)
        self.assertAlmostEqual(IQR, 3.)
        
        ## Test case 2: initial th_1 < th_2;
        th_1, th_2, upper_q, multiple, IQR = segment.compute_thresholds(self.array)
        
        self.assertAlmostEqual(th_1, 19.5)
        self.assertAlmostEqual(th_2, 20.)
        self.assertAlmostEqual(upper_q, 10.)
        self.assertAlmostEqual(multiple, 0.95)
        self.assertAlmostEqual(IQR, 10.)
        
    def test_threshold(self):
        
        ## Call function under test and pass generated data;
        seg_1, seg_2, seg_sum = segment.threshold(self.image,
                                                  self.mask,
                                                  self.th_1,
                                                  self.th_2)
        
        ## If the function output and the expected array are not equal, raise error;
        self.assertTrue( np.array_equal(self.seg_1, seg_1) )
        self.assertTrue( np.array_equal(self.seg_2, seg_2) )
        self.assertTrue( np.array_equal(self.seg_sum, seg_sum) )
        
    def test_remove_residues(self):
        
        ## Call function under test and pass generated data;
        seg = segment.remove_residues(self.seg_1, self.seg_sum)
        
        ## If the function output and the expected array are not equal, raise error;
        self.assertTrue( np.array_equal(self.seg, seg) )