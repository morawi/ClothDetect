#!/bin/bash
#SBATCH -p compute # which partition to run on ('compute' is default for two days, 'debug' for debugging, 'long' for 10 days )
#SBATCH --gres=gpu:rtx2080ti:1   # gpu:tesla:1 #  gpu:volta:1 # tesla:1 # gpu  # gpu:rtx2080ti:1
#SBATCH -J ModNeTst    # arbitrary name for the job (you choose)
#SBATCH --mem=20000    # how much RAM you need (30000 = 30GB in this case), if different from default; your job won't be able to use more than this
#SBATCH --cpus-per-task=2    # tell Slurm how many CPU cores you need, if different from default; your job won't be able to use more than this

# Uncomment the following to get a log of memory usage; NOTE don't use this if you plan to run multiple processes in your job and you are placing "wait" at the end of the job file, else Slurm won't be able to tell when your job is completed!
# vmstat -S M {interval_secs} >> memory_usage_$SLURM_JOBID.log &
 
# Your commands here
# nvidia-smi
python mask_rcnn.py --lr=0.005 --job_name="$SLURM_JOB_NAME" --lr_scheduler=StepLR --batch_size=2 --n_cpu=2 --num_epochs=1 --epoch=0 --checkpoint_interval=1 --HPC_run=True --redirect_std_to_file=True --pretrained_model=True --train_percentage=0.5 --dataset_name=Modanet

