# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:35:25 2020

@author: malrawi
"""

# from datasets import number_of_classes
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
parser.add_argument("--dataset_name", type=str, default="Modanet", help="name of the dataset: ClothCoParse or Modanet ")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--evaluate_interval", type=int, default=50, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
parser.add_argument("--train_percentage", type=float, default=0.9, help="percentage of samples used in training, the rest used for testing")
parser.add_argument("--experiment_name", type=str, default=None, help="name of the folder inside saved_models")
parser.add_argument("--print_freq", type=int, default=100, help="progress print out freq")
parser.add_argument("--lr_scheduler", type=str, default='OneCycleLR', help="lr scheduler name, one of: OneCycleLR, CyclicLR StepLR, ExponentialLR ")
parser.add_argument("--job_name", type=str, default='test', help=" name for the job used in slurm ")

parser.add_argument("--HPC_run", default=False, type=lambda x: (str(x).lower() == 'true'), help="True/False; -default False; set to True if running on HPC")
parser.add_argument("--remove_background", default=False, type=lambda x: (str(x).lower() == 'true'), help="True/False; - default False; set to True to remove background from image ")
parser.add_argument("--person_detection", default=False, type=lambda x: (str(x).lower() == 'true'), help=" True/False; - default is False;  if True will build a model to detect persons")
parser.add_argument("--train_shuffle", default=True, type=lambda x: (str(x).lower() == 'true'), help="True/False; -default True to shuffle training samples")
parser.add_argument("--redirect_std_to_file", default=False, type=lambda x: (str(x).lower() == 'true'),  help="True/False - default False; if True sets all console output to file")
parser.add_argument('--pretrained_model', default=True, type=lambda x: (str(x).lower() == 'true'), help="True/False: default True; True uses a pretrained model")


opt = parser.parse_args()
opt.num_epochs = opt.num_epochs+1 # to ensure generating and saving the last model, if any 
if platform.system()=='Windows':
    opt.n_cpu= 0

# # this used for debuging
# opt.pretrained_model=False
opt.batch_size = 1
opt.num_epochs = 1
opt.print_freq = 100
opt.checkpoint_interval=1
opt.train_percentage= 1 # 0.02 # to be used for debugging with low number of samples
# opt.lr_scheduler = 'StepLR'
# opt.person_detection=True

# opt.lr=0.01
# opt.checkpoint_interval=10
# opt.train_percentage=0.80 #0.02 # to be used for debugging with low number of samples
# opt.epoch=0
# opt.experiment_name = None # 'ClothCoParse-mask_rcnn-Mar-26-at-21-2'
# opt.sample_interval=5


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
data_loader, data_loader_test, num_classes = get_dataloaders(opt)

model = get_model_instance_segmentation( num_classes, pretrained_model=opt.pretrained_model )    
if opt.epoch != 0:
    # Load pretrained models
    print("loading model %s maskrcnn_%d.pth" % (opt.experiment_name, opt.epoch) )        
    model.load_state_dict(torch.load("saved_models/%s/maskrcnn_%d.pth" % (opt.experiment_name, opt.epoch)))
   
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
if opt.lr_scheduler=='StepLR':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
elif opt.lr_scheduler=='CyclicLR':
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.05)
elif opt.lr_scheduler=='OneCycleLR':
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=opt.num_epochs)


elif opt.lr_scheduler=='ExponentialLR':
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
else: print('Incorrect lr_scheduler')

# for epoch in range(opt.num_epochs):
for epoch in range(opt.epoch, opt.num_epochs):
    
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, 
                    opt.print_freq, lr_scheduler, opt)
    
        # Save model checkpoints
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval== 0:       
       print('Saving model ...')
       torch.save(model.state_dict(), "saved_models/%s/maskrcnn_%d.pth" % (opt.experiment_name, epoch))

    print('memory used:', torch.cuda.memory_stats(device))
    # # evaluate on the test dataset        
    # if epoch % opt.evaluate_interval == 0:
    #     evaluate(model, data_loader_test, device=device)
           

# sample_images(data_loader_test, model, device)

print("All done")

