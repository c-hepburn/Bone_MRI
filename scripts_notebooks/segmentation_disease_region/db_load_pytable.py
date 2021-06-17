import torch.utils.data
import tables

class LoadPyTable(torch.utils.data.Dataset):
    '''
    Purpose: load image and mask from a database.
    Input: path to the database file.
    Output:
    - image, 3D numpy array of size [1,H,W]
    - mask, 3D numpy array of size [1,H,W]
    
    Methods:
        __len__(): gives the total number of image/mask slices stored in the database;
        __getitem__(): retrieves an image and the corresponding mask, provided an index;
    '''    
    
    def __init__(self, path):
        '''
        Purpose: initialize class attributes.
        Attributes:
            - db: database file;
            - names: list, containing file root children names. By default, list includes 'Images'
            (array containing image slices), 'Masks' (array containing corresponding mask slices);
        '''
        ## Inheret from torch.utils.data.Dataset module
        super(LoadPyTable, self).__init__()
        
        ## Define attributes
        self.db = tables.open_file(path, mode = 'r')
        self.names = list(self.db.root._v_children.keys()) 
    
    def __len__(self):
        'Purpose: return total number of elements in the database.'
        name = self.names[0]
        return self.db.root[name].shape[0]
    
    def __getitem__(self, index):
        '''
        Purpose: return an image and a corresponding mask slice.
        Input: index
        Output:
            - image, 3D numpy array of size [1,H,W]
            - mask, 3D numpy array of size [1,H,W]
        '''
        images = self.db.root[self.names[0]] 
        masks = self.db.root[self.names[1]]
        
        ## Retrieve image and corresponding mask slices, provided an index
        image = images[index]
        mask = masks[index]
        return image, mask