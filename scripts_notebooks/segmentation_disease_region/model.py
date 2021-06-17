import torch
import torch.nn as nn
import torch.nn.functional as F
   
class UNet(nn.Module):
    '''
    Implementation of 2D Unet, "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    by O.Ronneberger et al., 2015;
    
    This implementation is taken from https://github.com/jvanvugt/pytorch-unet;
    Code modifications:
        - change in notation; added comments;
        - the order in which batch normalization is applied, i.e., before ReLU activation function;
        - no transpose convolution, only bilinear upsampling followed by 1x1 convolution;
        - ConvBlock and UpConvBlock modules are stored in pytorch sequential containers instead of
          a list;
        - new attribute: convolution kernel size;
        - attribute 'padding' takes integer values;
        - attribute 'output', boolean (if True output tensor size from each resolution layer is
          outputted);

    Default class attributes:
        - in_channels: 1, integer (number of input channels);
        - out_channels: 1, integer (number of output channels);
        - depth: 4, integer (affects number of resolution levels);
          Note: depth = 4 means that there are 4 down blocks in the contracting part and 3
          up blocks in the expansive part;
        - filter_number: 4, integer (determines the number of filters for each convolution layer;
          is given by 2^(filter_number+i), where i is in the range (0, depth - 1));
        - kernel_size: 3, integer (fixed for all convolution layers except that before bilinear
          upsampling and the last convolution layer, for which kernel size is 1);
        - padding: 1, integer (if kernel is a 5x5 matrix, padding must be 2 to keep the original
          tensor size, otherwise error will be raised);
        - batch_norm: True (if True, batch normalization is applied on the output of all
          convolution layers except that before bilinear upsampling and the last layer);
        - output: False;
    '''    
    def __init__(self,
        in_channels = 1,
        out_channels = 1,
        depth = 4,
        filter_number = 4,
        kernel_size = 3,
        padding = 1,
        batch_norm = True,
        output = False):
        '''
        Purpose:
        - initialize class attributes;
        - construct contracting, expansive paths, the last convolution layer;
 
        Note: this class calls the ConvBlock and UpConvBlock classes (below);
        '''
        ## Inherit from nn.Module;
        super(UNet, self).__init__()
        
        ## Initialize class attributes;
        self.prev_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.fn = filter_number
        self.kernel_size = kernel_size
        self.padding = padding
        self.batch_norm = batch_norm
        self.output = output
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()
        
        ## Construct the contracting path;
        #### ----------------------------------------- Loop over the levels of the contracting path; 
        for i in range(self.depth):
            
            self.down_path.append(
                ConvBlock(self.prev_channels,
                          2**(self.fn + i),
                          self.kernel_size,
                          self.padding,
                          self.batch_norm))
            
            ## Double number of the output channels with each level;
            self.prev_channels = 2**(self.fn + i) 
        #### --------------------------------------------------------------------------------------
        
        ## Construct the expansive path;
        #### ------------------------------------------ Loop over the levels of the expansive path;
        for i in reversed(range(depth - 1)): 
            
            self.up_path.append(
                UpConvBlock(self.prev_channels,
                            2**(self.fn + i),
                            self.kernel_size,
                            self.padding,
                            self.batch_norm))
            
            ## Halve number of the output channels with each level;
            self.prev_channels = 2**(self.fn + i) 
        #### --------------------------------------------------------------------------------------
        
        ## Construct the last convolution layer;
        self.last_layer = nn.Conv2d(self.prev_channels, self.out_channels, kernel_size = 1)
        
    def forward(self, image_batch):
        '''
        Purpose: pass the input tensor through the network;

        Input: image_batch, 4D pytorch tensor of size [N,C,H,W], where N - batch size, i.e., number
               of input image slices, C - number of input channels, H - input plane height and W -
               input plane width in pixels;
        Output: 4D pytorch tensor of size [N,C,H',W']; if padded, H'= H, W'= W;
        '''
        ## Initialize inputs;
        inputs = image_batch
        
        ## Create empty list to store the outputs from levels of the contracting path;
        output_storage = []

        #### --------------------------------------------- Loop over levels of the contracting path;
        for i, down_block in enumerate(self.down_path):
            
            ## Apply operations of the current level on the input tensor;
            output = down_block(inputs)
            
            ## If true, output current tensor size;
            if self.output:
                print('Down path, tensor size: ', output.shape)
            
            ## Save the output before the max pooling operation to concatenate with the outputs of
            ## the expansive path levels later;
            ## Note: max pooling is not performed on the output of the last level;
            ## ------------------------------------------------------------------ Condition on i
            if i != (self.depth-1): 
                
                ## Store the current output;
                output_storage.append(output)
                
                ## Apply max pooling operation;
                output = F.max_pool2d(output, kernel_size = 2, stride = 2)
                
                ## If true, output the current tensor size;
                if self.output:
                    print('Down path, max pooling, tensor size: ', output.shape)
                
                ## Re-assign the input tensor;
                inputs = output
            ## ---------------------------------------------------------------------------------- 

            ## Re-assign the input tensor;
            inputs = output
        #### ------------------------------------------------------------------------------------
        
        #### --------------------------------------------- Loop over levels of the expansive path;
        for i, up_block in enumerate(self.up_path):
            
            ## Apply operations of the current level on the input tensor;
            ## Concatenate with the corresponding tensor from the contracting path;
            output = up_block(inputs, output_storage[-i-1])
            
            ## If true, output current tensor size;
            if self.output:
                print('Up path, tensor size: ', output.shape)
            
            ## Re-assign the input tensor;
            inputs = output
        #### ------------------------------------------------------------------------------------
        
        ## Pass the output to the last convolution layer, followed by sigmoid activation function;
        output = self.last_layer(output)
        
        return torch.sigmoid(output)
        
class ConvBlock(nn.Module):
    '''
    Purpose:
        - create a block of several convolution layers, followed by batch normalization layer, if
          specified, and the activation function operator, ReLU;
        - apply operations of the block on the input;
        
    ConvBlock structure:
        - 2D convolution layer;
        - if True, batch normalization layer;
        - ReLU activation;
        - 2D convolutional layer;
        - if True, batch normalization;
        - ReLU activation;
    
    Input: 
        - image batch: 4D tensor of size [N,C,H,W];
        - number of input channels (integer);
        - number of output channels (integer);
        - kernel size (integer);
        - specify padding (boolean); if True, images are padded such as to maintain their original
          size after convolution;
        - batch_norm: boolean; if True, batch normalization is applied on the convolution layer
          output;
    
    Output: 4D torch tensor of size [N,C',H',W'];
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, batch_norm):
        '''
        Purpose: create a sequential container with modules (layers and operations);
        Input: 
            - number of input channels (integer);
            - number of output channels (integer);
            - kernel size (integer);
            - padding (boolean);
            - batch_norm (boolean);
        Output: sequential container;
        '''
        ## Inherit from nn.Module;
        super(ConvBlock, self).__init__() 
        
        ## Construct a sequential contrainer;
        if batch_norm: 
            self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
        else:
            self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding),
            nn.ReLU()
            )
            
    def forward(self, image_batch):
        '''
        Purpose: pass the input through all the layers/operations of the sequential container;
        Input: 4D torch tensor of size [N,C,H,W];
        Output: 4D torch tensor of size [N,C',H',W']; if padding True, H'= H, W'= W;
        '''
        return self.convblock(image_batch)
        
class UpConvBlock(nn.Module):
    '''
    Purpose: 
        - create a sequential container with modules (upsampling operation, 2D convolution layer);
        - call the ConvBlock class;
        - apply operations of the sequential container on the input;
        - concatenate the output with the corresponding cropped output from the contracting path
          ('skipped connection'); concatenation is along channel dimension;
        - pass the concatenated output through the ConvBlock;
    
    UpConvBlock structure:
        - Up Block;
        - ConvBlock (see above);
    
    UpBlock structure:
        - upsampling operator (bilinear interpolation);
        - 2D convolution layer;
    
    Input: 
        - 4D torch tensor of size [N,C,H,W];
        - 4D torch tensor of size [N,C//2,H',W'] of the corresponding level in the contracting path;
        - number of input channels (integer);
        - number of output channels (integer);
        - kernel size (integer);
        - padding (boolean);
        - batch_norm (boolean);

    Output: 4D torch tensor of size [N,C//2,H'',W'']; if padding True H''=2H, W''=2W;
    '''
    def __init__(self, in_channels, out_channels, kernel_size, padding, batch_norm):
        '''
        Purpose: create two sequential containers;
            - Container (1):
                - upsampling operator;
                - 2D convolutional layer;
            - Container (2): ConvBlock;
        '''
        ## Inherit from nn.Module;
        super(UpConvBlock, self).__init__()
    
        ## Construct sequential container with modules: upsamling, convolutional layer;
        self.UpBlock = nn.Sequential(
            nn.Upsample(mode = 'bilinear', scale_factor = 2, align_corners = True),
            nn.Conv2d(in_channels, out_channels, kernel_size = 1))
            
        ## Call ConvBlock class (described above);
        self.ConvBlock = ConvBlock(in_channels, out_channels, kernel_size, padding, batch_norm)
        
    def center_crop(self, image, target_shape):
        '''
        Purpose: center-crop tensor such that it matches the size of the target tensor;
        
        Input: 
            - 4D tensor of size [N,C,H,W] (to crop);
            - target tensor size: tuple (H',W');
            
        Output:
            - 4D tensor of size [N,C,H',W'] (center cropped);
        '''
        ## Retrieve input and target spatial dimensions;
        _, _, x, y = image.size() 
        tx, ty = target_shape
        
        ## Define start and stop values for cropping, i.e. how to slice the array;
        x_start, x_stop = x//2 - tx//2, x//2 + tx//2
        y_start, y_stop = y//2 - ty//2, y//2 + ty//2
        
        ## Crop the input;
        cropped = image[:, :, y_start : y_stop, x_start :  x_stop]
        
        ## Raise error if the cropped output size is different from the target size;
        assert(cropped.shape[2:] == target_shape)
        
        return cropped

    def forward(self, image, bridge):
        '''
        Purpose:
            - pass the input through UpBlock;
            - concatenate the output with the corresponding cropped output from the contracting
               path (skipped connection); concatenation is along channel dimension;
            - pass the output through ConvBlock;

        Input: 
            - 4D torch tensor of size [N,C,H,W];
            - 4D torch tensor of size [N,C//2,H',W'] (contracting path level output);

        Output: 4D torch tensor of size [N,C//2,H'',W'']; if padding True, H''=2H, W''=2W;
    
        Description of tensor size change: 
            Input to UpBlock: input tensor [N,C,H,W], bridge = [N,C//2,H',W'] 
                - UpBlock:
                    upsample       [N,C,H,W] ==> [N,C,2H,2W]
                    convolution    [N,C,2H,2W] ==> [N,C//2,2H,2W]
                    cropping bridge: [N,C//2,H',W'] ==> [N,C//2,2H,2W]
                    (if padding True, no actual cropping is performed);
                    concatenate along the channel axis (upsample+conv.,cropped) ==> [N,C,2H,2W] 

            Input to ConvBlock: [N,C,2H,2W]
                - ConvBlock:
                    if padding == False
                        convolution ==> [N,C//2,H',W'] 
                        convolution  ==> [N,C//2,H'',W''] 

                    if padding == True
                        convolution ==> [N,C//2,2H,2W] 
                        covolution  ==> [N,C//2,2H,2W] 
        '''
        ## Apply UpBlock;
        outputs = self.UpBlock(image)
        
        ## Crop the output from the corresponding contracting path level;
        cropped = self.center_crop(bridge, outputs.shape[2:])
        
        ## Concatenate along the channel dimension; 
        out = torch.cat([outputs, cropped], 1) 
        
        ## Apply ConvBlock;
        out = self.ConvBlock(out)
        
        return out