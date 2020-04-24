# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:14:25 2020

@author: malrawi
"""

import torchvision.transforms as transforms # torch transform used for computer vision applications
from PIL import Image
from datasets import ImageDataset, get_clothCoParse_class_names
from torch.utils.data import DataLoader
import torch
import utils
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from modanet_dataset import ModanetDataset
import sys


def get_any_image():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    # if  not filename:
    #     print("Program ended: Canceling selecting an image")
    #     sys.exit(0)
    return filename

def sample_images(images, targets, model, device, number_of_classes):   
    score_threshold = 0.8     
    class_name = get_clothCoParse_class_names() 
    images = list(image.to(device) for image in images)
    model.eval()  # setting model to evaluation mode
    with torch.no_grad():
        predictions = model(images)           # Returns predictions
    masks = predictions[0]['masks'].cpu().squeeze(1)
    labels = predictions[0]['labels'].cpu()
    scores = predictions[0]['scores'].cpu() # scores are already sorted
       
    np.set_printoptions(precision=2) 
    print("scores", scores.numpy())
    print('detected labels', labels.numpy())    
    print('sorted detected labels', labels.sort()[0].numpy())        
    print('original labels', targets[0]['labels'].sort()[0].numpy())
    print('detected labels', labels[scores>score_threshold].sort()[0].numpy(), '>scor_thr', score_threshold)
    print('-------------# # # #-----------')
       
    to_pil = transforms.ToPILImage()
    to_pil(images[0].cpu()).show()
    
       
    for i in range(len(labels)): # we have one label for each mask
        if scores[i].item()>score_threshold:
            Image.fromarray( 255*masks[i].numpy().round() ).show()
            print('label:', labels[i].item(), ', score: {:.2f}'.format(scores[i].numpy()), 'class is', class_name[labels[i]])
       
    model.train() # putting back the model into train status/mode 
    
def get_transforms():
    transforms_train = [
    # transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),    
    transforms.ToTensor(),
    transforms.Normalize( (.5, )*3, (.5, )*3, (.5, )*3),     ]    
   
    transforms_target = None
    
    transforms_test = [ transforms.ToTensor(),
    transforms.Normalize( (.5, )*3, (.5, )*3, (.5, )*3),    ]
    
    return transforms_train, transforms_test, transforms_target
    
def get_dataloaders(opt):
    
    # Configure dataloaders
    transforms_train, transforms_test, transforms_target = get_transforms()
    
    if opt.dataset_name=='Modanet':
        dataset=ModanetDataset("../data/%s" % opt.dataset_name, 
                             transforms_ = transforms_train,                             
                             HPC_run=opt.HPC_run, )
        dataset_test=ModanetDataset("../data/%s" % opt.dataset_name, 
                              transforms_ = transforms_test,                             
                              HPC_run=opt.HPC_run, )
       
    elif opt.dataset_name=='ClothCoParse':
                
        dataset = ImageDataset("../data/%s" % opt.dataset_name, 
                             transforms_ = transforms_train, 
                             transforms_target=transforms_target,
                             mode="train",                          
                             HPC_run=opt.HPC_run, 
                             remove_background = opt.remove_background,   
                             person_detection = opt.person_detection
                         )
        
        dataset_test = ImageDataset("../data/%s" % opt.dataset_name, 
                             transforms_ = transforms_test, 
                             transforms_target=transforms_target,
                             mode="train", # we are splitting data later, so all will be in train folder
                             HPC_run=opt.HPC_run, 
                             remove_background = opt.remove_background,   
                             person_detection = opt.person_detection
                         )
    else: 
        print( opt.dataset_name, " is incorrect dataset name, please select another one")
        sys.exit(0)
                    
    num_classes = dataset.number_of_classes(opt) # for some reason, number_of_classes will be lost if we move this line down after the block
    
    # split the dataset in train and test set
    train_samples = int(len(dataset)*opt.train_percentage)    
    dataset, dataset_test = torch.utils.data.random_split(dataset, [train_samples, len(dataset)-train_samples])
         
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=opt.train_shuffle,
        num_workers = opt.n_cpu,   
        collate_fn= utils.collate_fn
    )
    
       
    '''test is same as train for now'''
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
        collate_fn=utils.collate_fn
        
    )
    
    
    
    return dataloader, data_loader_test, num_classes
