# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:19:54 2020

@author: malrawi
"""

from datasets import number_of_classes
from models import get_model_instance_segmentation
from misc_utils import get_dataloaders, sample_images
import torch
from draw_image import instance_segmentation_api
# from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path2model", type=str, default=None, help="path to the model")
parser.add_argument("--model_name", type=str, default=None, help="name of the model")
parser.add_argument("--dataset_name", type=str, default="ClothCoParse", help="name of the dataset")
opt = parser.parse_args()

opt.model_name='maskrcnn_250.pth'
#opt.path2model= 'C:/MyPrograms/t-HPC-results/mask_rcnn-Mar-30-at-3-18/'
# opt.path2model ='C:/MyPrograms/t-HPC-results/mrcn_no_bkgrnd/'
opt.path2model ='C:/MyPrograms/t-HPC-results/mrcnn_prsn_detect/'
opt.HPC_run = 0
opt.remove_background = False
opt.person_detection = True
opt.train_percentage = 0.5
opt.batch_size = 1
opt.train_shuffle = False
opt.n_cpu=0
opt.cuda = False # this will definetly work on the cpu if it is false

folder_name = '/my'
image_name = 'myimage'

image_path = folder_name+image_name


device = torch.device('cuda' if opt.cuda else 'cpu')

model = get_model_instance_segmentation( number_of_classes(opt) )  
print("loading model", opt.model_name )        
model.load_state_dict(torch.load(opt.path2model+opt.model_name,  map_location=device ))  
model.eval()
instance_segmentation_api(model, folder_name+image_name)



    




