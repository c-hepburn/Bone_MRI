import torch

class Dice():
    '''
    Puprose: compute an approximation of the Dice coefficient (area overlap), averaged over
            - two classes (foreground, background);
            - number of prediction - (binary) reference standard slice pairs in the batch;

    ! Note: the code was written, taking into account the available data. Each reference standard
    slice contains pixels of both the foreground and background classes; zero dice for the
    foreground class, for a prediction - reference standard slice pair can result only from the
    model incorrect prediction;

    Dice score for a given class is implemented as (2 sum(r_i*p_i) / [sum(r_i^2) + sum(p_i^2)]),
    where summation runs over pixels of the reference standard, r_i and network probability map,
    p_i. "V-Net: fully convolutional neural networks for volumetric medical image segmentation",
    F. Milletari et al.; 

    Input:
        - p: network prediction, 4D torch tensor of size [N,1,H,W], allocated either on GPU or CPU;
        - r: reference standard, 4D torch tensor of size [N,1,H,W], allocated either on GPU or CPU;
    
    N - batch size, i.e., number of image slices, 1 - default number of input channels, H - input
    plane height and W - input plane width in pixels;
    
    Output: average dice score, i.e., area overlap (float);
    '''
    
    def compute_dice_scores(self, p, r):
        '''
        Purpose: compute approximate dice score for the foreground class, i.e., elements of the
                 reference standard (r or [1-r]) with the label 1;

        Inputs:
            - p: network prediction, 4D torch tensor of size [N,1,H,W];
            - r: reference standard (mask, binary image), 4D torch tensor of size [N,1,H,W];

        Output: 2D torch tensor of size [N,1], containing dice scores for each prediction -
                reference standard pair in the batch (N pairs);
        '''
        ## Compute number of the foreground class elements for each mask in the batch;
        ## Summation along H and W dimensions; output: 2D torch tensor of size [N,1];
        numel_r = torch.einsum('nchw->nc', [r**2])

        ## Compute approximate number of the foreground class elements for network predictions;
        ## Output: 2D torch tensor of size [N,1];
        numel_p = torch.einsum('nchw->nc', [p**2])

        ## Compute approximate number of the foreground class elements that have common location
        ## in the mask and network prediction;
        ## Element-wise multiplication is followed by summation along H, W dimensions;
        ## Output: 2D torch tensor of size [N,1]
        numel_intersection = torch.einsum('nchw,nchw->nc', [p, r])

        ## Compute dice scores for each mask-prediction pair;
        ## Output: 2D torch tensor of size [N,1];
        denominator = numel_r + numel_p
        scores = 2 * numel_intersection / denominator

        return scores

    def compute_average(self, tensor_1, tensor_2):
        '''
        Purpose: compute average of the elements of two tensors;

        Inputs:
        - 2D torch tensor of size [N,1] (dice scores for the foreground class);
        - 2D torch tensor of size [N,1] (dice scores for the background class);

        Output: 2D torch tensor of size [N,1];
        '''
        average = (tensor_1 + tensor_2) * 0.5

        return average

    def compute_average_elements(self, tensor):
        '''
        Purpose: compute average of the elements of a tensor;
        Input: 2D torch tensor of size [N,1];
        Output: average (float);
        '''
        average = torch.mean(tensor)

        return average.item()

    def compute_average_dice(self, p, r):
        '''
        Puprose: compute an approximation of Dice (area overlap), averaged over
                - two classes (foreground, background);
                - number of prediction-reference standard pairs in the batch;

        Input:
            - p: network prediction, 4D torch tensor of size [N,1,H,W];
            - r: reference standard, 4D torch tensor of size [N,1,H,W];

        Output: average dice score (float);
        '''
        ## Compute approximate dice scores (foreground class) for each mask-prediction pair;
        ## Output: 2D torch tensor of size [N,1];
        scores_FG = self.compute_dice_scores(p,r)

        ## Compute approximate dice scores (background class) for each mask-prediction pair;
        ## Output: 2D torch tensor of size [N,1];
        scores_BG = self.compute_dice_scores(1.-p, 1.-r)

        ## Compute approximate dice scores, averaged over the two classes;
        ## Output: 2D torch tensor of size [N,1];
        average_scores = self.compute_average(scores_FG, scores_BG)

        ## Compute average dice score (average over number of mask-prediction pairs);
        ## Output: float;
        average_dice = self.compute_average_elements(average_scores)

        return average_dice