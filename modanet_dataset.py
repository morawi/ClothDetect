# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:34:04 2020

@author: malrawi

Dataset class for Modanet dataset

"""


from torch.utils.data import Dataset # Dataset class from PyTorch
from PIL import Image# PIL is a nice Python Image Library that we can use to handle images
import torchvision.transforms as transforms # torch transform used for computer vision applications
import numpy as np
import torch
from pycocotools.coco import COCO

# idx_with_no_annotations = [12839, 12840, 12929, 12937, 12942, 13110, 13207 ,13214 ,18866 ,
#     18950 ,18978 ,18979 ,19023 ,19024 ,19062 ,19078 ,19107 ,19139 ,
#     19143 ,19193 ,19197 ,19214 ,23715 ,23742 ,23755 ,23880 ,24089 ,
#     24238 ,24284 ,24372 ,24483 ,24657 ,24786 ,24959 ,25094 ,25149 ,
#     25300 ,25323 ,25356 ,25413 ,25426 ,25651 ,25703 ,25734 ,25777 ,
#     25915 ,25940 ,26143 ,26177 ,26252 ,26285 ,26327 ,26341 ,26522 ,
#     26597 ,26705 ,26802 ,26900 ,27041 ,27043 ,27087 ,27112 ,27122 ,
#     27338 ,27360 ,27364 ,27384 ,27427 ,27440 ,27472 ,27542 ,27798 ,
#     27800 ,27826 ,28081 ,28230 ,28503 ,28557 ,28595 ,28667 ,28669 ,
#     28858 ,28899 ,29027 ,29348 ,35490 ,35539 ,35542 ,35546 ,35576 ,
#     35657 ,35679 ,35711 ,35770 ,35909 ,35971 ,35989 ,36011 ,36014 ,
#     36080 ,36123 ,36139 ,36149 ,36150 ,36199 ,36208 ,36211 ,36352 ,
#     36361 ,36385 ,36408 ,36444 ,36456 ,36468 ,36514 ,36522 ,36570 ,
#     36576 ,36592 ,36624 ,36644 ,36728 ,36754]
# wrong

idx_with_no_annotations= [1365 ,1388 ,453445 ,456009 ,456939 ,1089041,461792,462919,
 372190 ,375419,376816 ,376893 ,379849 ,379885 ,382041 ,382689 ,384182 ,385450 ,385655 ,
 387901,388069 ,388888 ,391213 ,391662 ,391888 ,393778 ,396963 ,399280 ,400270 ,402167 ,
 404415 ,407472 ,409988 ,412324 ,413862 ,414502 ,416904 ,417129 ,417501 ,418273 ,
 418456 ,421172 ,421835 ,422213 ,422890 ,424473 ,424747 ,426923 ,427335 ,428161 ,
 428537 ,429048 ,429197 ,431241 ,432115 ,433070 ,434238 ,435414 ,437598 ,437625 ,
 438183 ,438479 ,438647 ,441740 ,442059 ,442085 ,442390 ,443352 ,443504 ,443895 ,
 444988 ,448831 ,448833 ,449084 ,452479 ,454538 ,458468 ,459503 ,460094 ,461135 ,
 461160 ,464158 ,464833 ,467034 ,471426 ,473854 ,476166 ,476419 ,476494 ,477518 ,
 480370 ,481208 ,482374 ,484909 ,491356 ,493922 ,494724 ,495620 ,495868 ,498200 ,
 499765 ,500395 ,500690 ,500773 ,502342 ,502656 ,502792 ,507985 ,508391 ,509054 ,
 509775 ,511635 ,512043 ,513044 ,515331 ,515526 ,517350 ,517759 ,518488 ,520086 ,
 521059 ,524382 ,525684]
gray_level_imgs=[796310 ,288740 ,163205 ,648856 ,972996 ,397256 ,418664 ,419862 ,596299]
other_images = [453049, 161805]
take_out_images = idx_with_no_annotations + gray_level_imgs + other_images


class ModanetDataset(Dataset):
    def __init__(self, root='C:/MyPrograms/Data/Modanet', 
                 transforms_=None, HPC_run=False ):
        
        if HPC_run:
            root = '/home/malrawi/MyPrograms/Data/Modanet'        
                        
        if transforms_ != None:
            self.transforms = transforms.Compose(transforms_) # image transform
        else: self.transforms=None     
        
        self.annFile = root + '/modanet2018_instances_train.json'
        self.path2images = root + '/images/'

        self.coco=COCO(self.annFile)
        # COCO categories and supercategories
        self.cat_names = self.coco.loadCats(self.coco.getCatIds()) # class names       
        self.catIds = self.coco.getCatIds(catNms=['']) # using all categories
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
      
    
    def __getitem__(self, index):    
        img_file = self.coco.loadImgs(self.imgIds[index] )[0]
        while img_file['id'] in take_out_images: 
            print('random replace ann ', img_file['id'], 'as it does not exist')
            index = torch.randint(0, len(self.imgIds)-1, (1,)) # generarte a random index to replace that one
            img_file = self.coco.loadImgs(self.imgIds[index] )[0]
                       
        image_A = Image.open(self.path2images + img_file['file_name'])        
        annIds = self.coco.getAnnIds(imgIds=img_file['id'], catIds=self.catIds, 
                                     iscrowd=None) # suppose all instances are not crowd        
        
        anns = self.coco.loadAnns(annIds)   
        num_objs=len(anns)
        
        
        boxes=[]; labels=[]; area=[]               
        masks = np.zeros((num_objs, img_file['height'], img_file['width'] ) ) # just getting the shape of the mask
        for i in range(num_objs):
            labels.append(anns[i]['category_id'])            
            masks[i,:,:] = self.coco.annToMask(anns[i])            
            # boxes.append(anns[i]['bbox']) # seems there is a problem Modanet boxes 
            # area.append(anns[i]['area']) # and areas, not sure if it is in all images
              
        '''  I am calculating the bboxes and areas from the masks
             as I think they are incorrect, I've had a nan in the maskrcnn loss, 
             then after checking, the area does not conform with the bounding boxes        
             the same maskrcnn worked very well on the clothingCoParse dataset
        '''
        # if min(labels)<1 or 
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])        
      
        boxes = torch.as_tensor(boxes, dtype=torch.float32)      
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {}                
        target["boxes"]=  boxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) 
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        target["image_id"] = torch.tensor([index]) # or, should it be this one? img_file['id'], the tutorial shows that this is the index https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 
        target["area"] =  area
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64) # suppose all instances are not crowd             

        if self.transforms != None:
            image_A = self.transforms(image_A)                
        
        return image_A, target
     

    def __len__(self): # this function returns the length of the dataset, the source might not equal the target if the data is unaligned
        return len(self.imgIds)
    
    def number_of_classes(self, opt):
        return(len(self.cat_names)+1) # this should do

# transforms_ = [
#     transforms.Resize((300, 300), Image.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]

x_data = ModanetDataset(transforms_=None)
# im, tg = x_data[0] # [12839]
for idx in range(len(x_data)):
    # print(idx,',', end='')
    x_data[idx]
    
    
'''  used to remove problematic samples
    # if image_A.mode !='RGB': # num_objs<1
        #     print(img_file['id'],',', end='')            
        # return 

'''


