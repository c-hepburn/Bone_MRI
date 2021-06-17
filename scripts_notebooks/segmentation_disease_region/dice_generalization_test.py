import unittest
from dice_generalization import *

## Create Dice class object;
dice = Dice()

class SoftDiceTest(unittest.TestCase):
    
    def setUp(self):
        'Purpose: generate inputs and the expected outputs to test Dice class methods;'
        
        ## Create 4D torch tensor of size [2,1,3,3], that represents a (binary) referense
        ## standard (rs);
        rs = torch.zeros([2,1,3,3])
        rs[0,0,:,0], rs[0,0,:2,1], rs[1,0,:,0] = 1., 1., 1.

        ## Create 4D torch tensor of size [2,1,3,3], that represents the corresponding model
        ## prediction;
        p = torch.zeros([2,1,3,3])
        p[0,0,:,0] = 0.8
        p[0,0,:,1] = 0.1
        p[0,0,0,2] = 0.9
        p[1,0,:,0] = 0.7
        p[1,0,:,1] = 0.2
        
        ## Expected output: scores for the foreground and background classes for each
        ## prediction - rs slice pair;
        scores_FG = torch.tensor([[0.6701],[0.9150]])
        scores_BG = torch.tensor([[0.7009],[0.9651]])
        
        ## Expected output: average of the two tensors;
        average = torch.tensor([[0.6855],[0.9401]])
        
        ## Expected output: average dice;
        average_dice = 0.8128
        
        self.rs = rs
        self.p = p
        self.scores_FG = scores_FG
        self.scores_BG = scores_BG
        self.average = average
        self.average_dice = average_dice
        
    def test_compute_dice_scores(self):
        
        ## Compute dice scores for the foreground class;
        scores_FG = dice.compute_dice_scores(self.p, self.rs)
        
        ## Compute dice scores for the background class;
        scores_BG = dice.compute_dice_scores(1. - self.p, 1. - self.rs)
        
        ## Check whether the result matches the expected output;
        self.assertTrue( torch.allclose(scores_FG, self.scores_FG, rtol=1e-04, atol=1e-08))
        self.assertTrue( torch.allclose(scores_BG, self.scores_BG, rtol=1e-04, atol=1e-08))
        
    def test_compute_average(self):
        
        ## Compute average of the elements of two tensors;
        average = dice.compute_average(self.scores_FG, self.scores_BG)
        
        ## Check whether the result matches the expected output;
        self.assertTrue( torch.allclose(average, self.average, rtol=1e-04, atol=1e-08))
        
    def test_compute_average_dice(self):
        
        ## Compute approximation of Dice, averaged over classes and prediction-rs slice pairs;
        average_dice = dice.compute_average_dice(self.p, self.rs)

        ## Check whether the result matches the expected output (up to 4th digit);
        self.assertAlmostEqual(average_dice, self.average_dice, places = 4)