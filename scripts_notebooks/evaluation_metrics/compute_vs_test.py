import unittest
from compute_vs import *

class ComputeVSTest(unittest.TestCase):
    
    def setUp(self):
        'Purpose: generate input and the expected output to test compute_vs function;'
        
        ## Generate input data;
        array_1 = np.zeros([3,3,2])
        array_1[:,0,0], array_1[0,:,1] = 1., 1.

        array_2 = np.zeros([3,3,2])
        array_2[:,1,0], array_2[0,0,0] = 1., 1.

        self.array_1 = array_1
        self.array_2 = array_2
        
    def test_compute_overlap(self):
        '''Three considered cases:
        - V1 > V2
        - V1 < V2
        - V1 = V2
        (Note: if V1 = V2 = 0, error is raised by the function);'''
        
        ## Output: tuple (vs, voxel count in the first array, voxel count in the 2nd array);
        output_1 = compute_vs(self.array_1, self.array_2)
        output_2 = compute_vs(self.array_2, self.array_1)
        output_3 = compute_vs(self.array_1, self.array_1)
        
        self.assertAlmostEqual(output_1[0], 0.8)
        self.assertAlmostEqual(output_1[1], 6.)
        self.assertAlmostEqual(output_1[2], 4.)
        self.assertAlmostEqual(output_2[0], 0.8)
        self.assertAlmostEqual(output_2[1], 4.)
        self.assertAlmostEqual(output_2[2], 6.)
        self.assertAlmostEqual(output_3[0], 1.)
        self.assertAlmostEqual(output_3[1], 6.)
        self.assertAlmostEqual(output_3[2], 6.)