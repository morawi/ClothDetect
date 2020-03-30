# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:19:54 2020

@author: malrawi
"""

from datasets import number_of_classes
from models import get_model_instance_segmentation
from misc_utils import get_dataloaders, sample_images
import torch
# from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path2model", type=str, default=None, help="path to the model")
parser.add_argument("--model_name", type=str, default=None, help="name of the model")
parser.add_argument("--dataset_name", type=str, default="ClothCoParse", help="name of the dataset")
opt = parser.parse_args()

opt.model_name='maskrcnn_250.pth'
opt.path2model= 'C:/MyPrograms/t-HPC-results/mask_rcnn-Mar-30-at-3-18/'
opt.HPC_run = 0
opt.remove_background = 1
opt.person_detection = 0
opt.train_percentage = 0.5
opt.batch_size = 1
opt.train_shuffle = 0
opt.n_cpu=0
opt.cuda = False # this will definetly work on the cpu if it is false

device = torch.device('cuda' if opt.cuda else 'cpu')
model = get_model_instance_segmentation( number_of_classes(opt.dataset_name) )  
print("loading model", opt.model_name )        
model.load_state_dict(torch.load(opt.path2model+opt.model_name,  map_location=device ))  
model.eval()
data_loader, data_loader_test = get_dataloaders(opt)

for i in range(3): # range( len(data_loader_test)): # let's just check a couple of images
    images, targets = next(iter(data_loader_test)) # get image(s)
    sample_images(images, targets, model, device)

    




