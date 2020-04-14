# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:19:54 2020

@author: malrawi
"""

from datasets import number_of_classes
from models import get_model_instance_segmentation
from misc_utils import get_dataloaders, sample_images, get_any_image
import torch
import argparse
from draw_image import instance_segmentation_api


parser = argparse.ArgumentParser()
parser.add_argument("--path2model", type=str, default=None, help="path to the model")
parser.add_argument("--model_name", type=str, default=None, help="name of the model")
parser.add_argument("--dataset_name", type=str, default="ClothCoParse", help="name of the dataset")
opt = parser.parse_args()

opt.model_name='maskrcnn_300.pth'

#opt.path2model= 'C:/MyPrograms/t-HPC-results/Maskrcnn-April-2/mask_rcnn-Apr-1-at-16-31/' # person detect
# opt.path2model= 'C:/MyPrograms/t-HPC-results/Maskrcnn-April-2/mask_rcnn-Apr-1-at-16-10/' # no pre train cloth segment
#opt.path2model= 'C:/MyPrograms/t-HPC-results/Maskrcnn-April-2/mask_rcnn-Apr-1-at-16-34/' # pre train cloth segment
opt.path2model = 'C:/MyPrograms/mask_rcnn-Apr-3-at-16-28/'
opt.path2model = 'C:/MyPrograms/saved_models/ClothCoParse/mask_rcnn-Apr-4-at-15-45/' # keep background

opt.person_detection = False

opt.HPC_run = 0
opt.remove_background = True
opt.train_percentage = 0.5
opt.batch_size = 1
opt.train_shuffle = 0
opt.n_cpu=0
opt.cuda = True # this will definetly work on the cpu if it is false

opt.load_via_GUI = True

device = torch.device('cuda' if opt.cuda else 'cpu')
model = get_model_instance_segmentation( number_of_classes(opt) )  
print("loading model", opt.model_name )        
model.load_state_dict(torch.load(opt.path2model+opt.model_name,  map_location=device ))  
model.to(device)
model.eval()
data_loader, data_loader_test = get_dataloaders(opt)

if opt.load_via_GUI:
    image_name = get_any_image()
    instance_segmentation_api(model, image_name, device, threshold=0.7, rect_th=1, text_size=1, text_th=3)

else:
    for i, batch in enumerate(data_loader_test): # let's just check a couple of images
        if i>=1: break
        images, targets = batch # get image(s)
        sample_images(images, targets, model, device, number_of_classes(opt))
     
'''
# saving segmented cloths
def save_masks_as_images(img, masks, path, file_name, labels)
for i in range(len(masks)):
    image_A = ImageChops.multiply(img, Image.fromarray(255*masks[i]).convert('RGB') )
    image_A.save(path+file_name+labels[i]+'.png')
    
'''



