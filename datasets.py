import glob
import os
import scipy.io as sio
from torch.utils.data import Dataset # Dataset class from PyTorch
from PIL import Image, ImageChops # PIL is a nice Python Image Library that we can use to handle images
import torchvision.transforms as transforms # torch transform used for computer vision applications
import numpy as np
import torch
import sys

# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


def get_clothCoParse_class_names(): 
    # names ordered according to label id, 0 for background and 59 for wedges
    
    ClothCoParse_class_names = ['background',  'accessories',  'bag',  'belt',  'blazer',
 'blouse',  'bodysuit',  'boots',  'bra',  'bracelet',  'cape',  'cardigan',
 'clogs', 'coat',  'dress', 'earrings', 'flats', 'glasses', 'gloves', 'hair',
 'hat', 'heels', 'hoodie', 'intimate', 'jacket', 'jeans', 'jumper', 'leggings',
 'loafers', 'necklace', 'panties', 'pants', 'pumps', 'purse', 'ring', 'romper',
 'sandals', 'scarf', 'shirt', 'shoes', 'shorts', 'skin', 'skirt', 'sneakers',
 'socks', 'stockings', 'suit', 'sunglasses', 'sweater', 'sweatshirt', 'swimwear',
 't-shirt', 'tie', 'tights', 'top', 'vest', 'wallet', 'watch', 'wedges']
    
    return ClothCoParse_class_names
    

def number_of_classes(opt):
    if opt.person_detection:
        return 2
    else:
        return(len(get_clothCoParse_class_names())) # this should do
        # return (59 + 1) # should be 59 in total, will change it later


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, transforms_target=None,
                 mode="train", person_detection=False,
                 HPC_run=False, remove_background=True):
        
        self.remove_background = remove_background # we'll have to add it as an argument later
        self.person_detection =person_detection
        
        if transforms_ != None:
            self.transforms = transforms.Compose(transforms_) # image transform
        else: self.transforms=None
        if transforms_target != None:
            self.transforms_target = transforms.Compose(transforms_target) # image transform
        else: self.transforms_target=None
        
        
        if HPC_run:
            root = '/home/malrawi/MyPrograms/Data/ClothCoParse'
        
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*")) # get the source image file-names
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*")) # get the target image file-names
      

    def __getitem__(self, index):              
                
        annot = sio.loadmat(self.files_B[index % len(self.files_B)])
        mask = annot["groundtruth"]
        image_A = Image.open(self.files_A[index % len(self.files_A)]) # read the image, according to the file name, index select which image to read; index=1 means get the first image in the list self.files_A

            
        if self.remove_background or self.person_detection: 
            mm = np.int8(mask>0) # thresholding the mask            
            image_A = ImageChops.multiply(image_A, Image.fromarray(255*mm).convert('RGB') )
            if self.person_detection:
                mask = mm # this is a binary mask
                        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)[1:] # first id is the background, so remove it
        
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
                
        
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64) # original, works only for two class problem
        labels = torch.as_tensor(obj_ids, dtype=torch.int64) # corrected by Rawi
                
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms != None:
            img = self.transforms(image_A)
        if self.transforms_target != None:
            target = self.transforms_target(target)
            
        
        return img, target
     

    def __len__(self): # this function returns the length of the dataset, the source might not equal the target if the data is unaligned
        return len(self.files_B)

# transforms_ = [
#     transforms.Resize((300, 300), Image.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]


# x_data = ImageDataset("../data/%s" % "ClothCoParse",  
#                             transforms_= '', #transforms_,                            
#                             mode = "train",   
#                             HPC_run = False,
#                             )

# for i in range(len(x_data)):
#     print(i)
#     z= x_data[i]  #accessing the first element in the data, should have the first image and its corresponding pixel-levele annotation

# x_data[0][1]

# # plt.imshow(anno.convert('L'),  cmap= plt.cm.get_cmap("gist_stern"), vmin=0, vmax=255)


# if num_objs==0:     # this can/should be used to (data cleaning) remove pairs with no annotations as these will cause an error       
        #     print('############ 0 objects ################# ')
        #     print(self.files_B[index % len(self.files_B)])
        