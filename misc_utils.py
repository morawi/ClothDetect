# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:14:25 2020

@author: malrawi
"""

import torchvision.transforms as transforms # torch transform used for computer vision applications
from PIL import Image
from datasets import ImageDataset
from torch.utils.data import DataLoader
import torch
import utils


def sample_images(images, targets, model, device, number_of_classes):    
    # Image.fromarray(255*targets[0]['masks'].squeeze(0).numpy()).show()
    images = list(image.to(device) for image in images)
    model.eval()  # setting model to evaluation mode
    with torch.no_grad():
        predictions = model(images)           # Returns predictions
    masks = predictions[0]['masks'].cpu().squeeze(1)
    labels = predictions[0]['labels'].cpu()
    scores = predictions[0]['scores'].cpu() # scores are already sorted
    model.train() # putting back the model into train status/mode 
    
    
    print(labels)
    print(targets[0]['labels'])
    print('-------------####-----------')
       
    for i in range(number_of_classes-1): # we have one label for each mask
         Image.fromarray( 255*masks[i].numpy().round() ).show()
         print('label:', labels[i], ', score:', scores[i])
    to_pil = transforms.ToPILImage()
    to_pil(images[0].cpu()).show()
    

def get_dataloaders(opt):
    # Configure dataloaders
    transforms_train = [
        # transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize( (.5, )*3, (.5, )*3, (.5, )*3),
    ]
    
   
    transforms_target = None
    
    
    transforms_test = [ transforms.ToTensor(),
    transforms.Normalize( (.5, )*3, (.5, )*3, (.5, )*3),
    ]
    
    
    
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
    
     # split the dataset in train and test set
    train_samples = int(len(dataset)*opt.train_percentage)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:train_samples])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[train_samples:])

   
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
    
    return dataloader, data_loader_test
