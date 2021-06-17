import unittest
from compute_dice import *

class ComputeDiceTest(unittest.TestCase):
    
    def setUp(self):
        'Purpose: generate input and the expected output to test compute_overlap function;'
        
        ## Generate input data;
        array_1 = np.zeros([3,3,2])
        array_1[0,0,0], array_1[:,1,0] = 1., 1.
        array_1[:,2,1] = 1.
        
        array_2 = np.zeros([3,3,2])
        array_2[1,0,0], array_2[1:,1,0] = 1., 1.
        array_2[:,1,1], array_2[:,2,1] = 1., 1.
        
        self.array_1 = array_1
        self.array_2 = array_2
        self.array_3 = np.zeros([3,3,2])
        
    def test_compute_overlap(self):
        '''Four considered cases:
        - perfect overlap (3D);
        - partial overlap (3D);
        - partial overlap (2D);
        - zero overlap due to the absence of the foreground elements in one array (3D);'''
        
        output_1 = compute_overlap(self.array_1, self.array_1)
        output_2 = compute_overlap(self.array_1, self.array_2)
        output_3 = compute_overlap(self.array_1[:,:,0], self.array_2[:,:,0])
        output_4 = compute_overlap(self.array_1, self.array_3)
        
        self.assertAlmostEqual(output_1, 1.)
        self.assertAlmostEqual(output_2, 0.625)
        self.assertAlmostEqual(output_3, 0.5714, places = 4)## (up to 4th digit)
        self.assertAlmostEqual(output_4, 0.)