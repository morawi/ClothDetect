# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:35:25 2020

@author: malrawi
"""

# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from engine import train_one_epoch, evaluate
import torch
from datasets import number_of_classes
from models import get_model_instance_segmentation
from misc_utils import get_dataloaders
import argparse
import platform
from engine import train_one_epoch, evaluate
import datetime
import calendar
import os


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="ClothCoParse", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type= int, default=512, help="size of image width")

parser.add_argument("--sample_interval", type=int, default=100, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
parser.add_argument("--HPC_run", type=int, default=0, help="if 1, sets to true if running on HPC: default is 0 which reads to False")
parser.add_argument("--remove_background", type=int, default=0, help="if 1, sets to true if: default is 1 which reads to False")
parser.add_argument("--redirect_std_to_file", type =int, default=0, help="set all console output to file: default is 0 which reads to False")
parser.add_argument("--train_percentage", type=float, default=0.8, help="percentage of samples used in training, the rest used for testing")


opt = parser.parse_args()

if platform.system()=='Windows':
    opt.n_cpu= 0

opt.train_shuffle = False
opt.batch_size = 2
opt.num_epochs = 2
opt.print_freq = 10
opt.checkpoint_interval=1
opt.train_percentage=0.02 # to be used for debugging with low number of samples

if platform.system()=='Windows':
    opt.n_cpu= 0

print(opt)

dt = datetime.datetime.today()
opt.experiment_name = opt.dataset_name+"-mask_rcnn-"+calendar.month_abbr[dt.month]+"-"+str(dt.day)+'-at-'+str(dt.hour) +'-'+str(dt.minute)

os.makedirs("images/%s" % opt.experiment_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.experiment_name, exist_ok=True)



# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# use our dataset and defined transformations

data_loader, data_loader_test = get_dataloaders(opt)

model = get_model_instance_segmentation( number_of_classes(opt.dataset_name) )    
if opt.epoch != 0:
    # Load pretrained models
    model.load_state_dict(torch.load("saved_models/%s/maskrcnn__%d.pth" % (opt.dataset_name, opt.epoch)))
   
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
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
    
    evaluate(model, data_loader, device=device) # used for debugging
    
    # evaluate(model, data_loader_test, device=device)
    
    # evaluate(model, data_loader, device=device) # just for debugging
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval== 0:
       # Save model checkpoints
       torch.save(model.state_dict(), "saved_models/%s/maskrcnn_%d.pth" % (opt.experiment_name, epoch))
       


print("All done")

# images,targets = next(iter(data_loader))
# device = torch.device('cpu')
# # images= images.to(device)
# images = list(image.to(device) for image in images)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
# output = model(images,targets)   # Returns losses and detections
# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)           # Returns predictions


