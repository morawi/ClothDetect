# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:35:25 2020

@author: malrawi
"""

from datasets import number_of_classes
from models import get_model_instance_segmentation
from engine import train_one_epoch, evaluate
from misc_utils import get_dataloaders
import torch
import argparse
import platform
import datetime
import calendar
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--num_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="ClothCoParse", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.005, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type= int, default=512, help="size of image width")
parser.add_argument("--evaluate_interval", type=int, default=50, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
parser.add_argument("--HPC_run", type=int, default=0, help="if 1, sets to true if running on HPC: default is 0 which reads to False")
parser.add_argument("--remove_background", type=int, default=0, help="if 1, sets to true if: default is 1 which reads to False")
parser.add_argument("--redirect_std_to_file", type =int, default=0, help="set all console output to file: default is 0 which reads to False")
parser.add_argument("--train_percentage", type=float, default=0.8, help="percentage of samples used in training, the rest used for testing")
parser.add_argument("--experiment_name", type=str, default=None, help="name of the folder inside saved_models")
parser.add_argument("--print_freq", type=int, default=100, help="progress print out freq")

opt = parser.parse_args()
opt.train_shuffle = True 
if platform.system()=='Windows':
    opt.n_cpu= 0


# this used for debuging
opt.batch_size = 2
# opt.num_epochs = 11
# opt.print_freq = 10
# opt.checkpoint_interval=10
# opt.train_percentage=0.80 #0.02 # to be used for debugging with low number of samples
# opt.epoch=0
# opt.experiment_name = None # 'ClothCoParse-mask_rcnn-Mar-26-at-21-2'
# opt.sample_interval=5

def sample_images(data_loader_test, model, device):
    images,targets = next(iter(data_loader_test)) # grab the images
    images = list(image.to(device) for image in images)
    model.eval()  # setting model to evaluation mode
    with torch.no_grad():
        predictions = model(images)           # Returns predictions
    masks = predictions[0]['masks'].cpu().squeeze(1)
    labels = predictions[0]['labels'].cpu()
    model.train() # putting back the model into train status/mode 
    
    
    ''' TODO: Do something with the predictions, ie display / save '''
    



# sanity check
if opt.epoch !=0 and opt.experiment_name is None:
    print('When epohch is not 0, there should be the name of the folder that has previously stored model')
    sys.exit()
elif opt.epoch !=0 and opt.experiment_name is not None:
    opt.experiment_name = opt.dataset_name +'/'+ opt.experiment_name
else: # totaly new experiment        
    dt = datetime.datetime.today()
    opt.experiment_name = opt.dataset_name+'/'+"mask_rcnn-"+calendar.month_abbr[dt.month]+"-"+str(dt.day)+'-at-'+str(dt.hour) +'-'+str(dt.minute)
    os.makedirs("images/%s" % opt.experiment_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.experiment_name, exist_ok=True)

if  opt.redirect_std_to_file:    
    out_file_name = "saved_models/%s" % opt.experiment_name
    print('Output sent to ', out_file_name)
    sys.stdout = open(out_file_name+'.txt',  'w')

print(opt)


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# use our dataset and defined transformations

data_loader, data_loader_test = get_dataloaders(opt)

model = get_model_instance_segmentation( number_of_classes(opt.dataset_name) )    
if opt.epoch != 0:
    # Load pretrained models
    print("loading model %s maskrcnn_%d.pth" % (opt.experiment_name, opt.epoch) )        
    model.load_state_dict(torch.load("saved_models/%s/maskrcnn_%d.pth" % (opt.experiment_name, opt.epoch)))
   
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


for epoch in range(opt.num_epochs):
    
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=opt.print_freq)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset    
    
    if epoch % opt.evaluate_interval == 0:
        evaluate(model, data_loader_test, device=device)
        
    
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval== 0:
       # Save model checkpoints
       print('Saving model ...')
       torch.save(model.state_dict(), "saved_models/%s/maskrcnn_%d.pth" % (opt.experiment_name, epoch))
       

# sample_images(data_loader_test, model, device)

print("All done")



# images,targets = next(iter(data_loader_test))
# images = list(image.to(device) for image in images)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
# output = model(images,targets)   # Returns losses and detections
# # For inference
# model.eval()  # probably not needed here 
# predictions = model(images)
