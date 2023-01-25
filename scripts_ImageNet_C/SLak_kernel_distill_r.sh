#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o ./Slak_Robustness_imagenet_r_distillation.out

source /home/sliu/miniconda3/etc/profile.d/conda.sh
#source activate slak

#python -m torch.distributed.launch --nproc_per_node=4 main.py  \
#--resume ./checkpoints/SLaK_tiny_checkpoint.pth --Decom True  --width_factor 1.3 -u 2000 --epochs 300 --model SLaK_tiny --drop_path 0.1 --batch_size 128 --lr 4e-3 --update_freq 8 --model_ema false --model_ema_eval true \
#--data_path /projects/2/managed_datasets/imagenet/ --num_workers 40 \
#--kernel_size 51 49 47 13 5 --output_dir ./
#/home/sliu/project_space/datasets/
data=/home/sliu/project_space/datasets/imagenet-r
data_type=imagenet-r
##python  main.py --model SLaK_tiny --kernel_size 3 3 3 3 100 --width_factor 1.0 --eval True --Decom False  --resume /projects/0/prjste21060/projects/dino/dino_slak_3_bn/linear/checkpoint.pth.tar --input_size 224 --drop_path 0.1 --data_path $data
#python  main.py --model SLaK_tiny  --kernel_size 5 5 5 5 100 --width_factor 1.0 --eval True --Decom False  --resume /home/sliu/project_space/datasets/checkpoints/5x5/checkpoint-best.pth --input_size 224 --drop_path 0.1 --data_path $data
#python main.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --Decom False  --resume /home/sliu/project_space/datasets/checkpoints/ConvNext/checkpoint-best.pth --input_size 224 --drop_path 0.1 --data_path $data 
#python main.py --model SLaK_tiny --eval True  --kernel_size 31 29 27 13 5 --width_factor 1.0 --Decom False --resume /home/sliu/project_space/datasets/checkpoints/rep/checkpoint-best.pth --input_size 224 --drop_path 0.1 --data_path $data
#python -m torch.distributed.launch --nproc_per_node=4 main.py --model SLaK_tiny  --kernel_size 51 49 47 13 5 --width_factor 1.3 --Decom True -u 2000 --sparsity 0.4  --sparse_init snip  --prune_rate 0.5 --growth random --resume /home/sliu/project_space/datasets/checkpoints/slak/SLaK-T-51-checkpoint-best.pth --input_size 224 --drop_path 0.1 --lr 1e-3 --update_freq 8 --data_path /projects/2/managed_datasets/imagenet/
#python main.py --model SLaK_tiny --eval true --kernel_size 51 49 47 13 5 --width_factor 1.3 --Decom True  --resume /home/sliu/project_space/datasets/checkpoints/slak/SLaK-T-51-checkpoint-best.pth --input_size 224 --drop_path 0.1 --data_path $data
# ck=/projects/0/prjste21060/projects/LoRA_LK/transfer/SLAK_Attention_tiny/514947135/vertical_kernels_with_rep/checkpoint-best.pth
# python main.py  --LoRA True --epochs 120 --model SLAK_Attention_tiny --drop_path 0.1 --batch_size 128 --lr 4e-3 --resume $ck --update_freq 8 --model_ema true \
# --model_ema_eval true --data_path $data --num_workers 18 --kernel-size 51 49 47 13 5 --bn True

#distillation
# ck1=/home/sliu/project_space/datasets/T05/checkpoint-best.pth
# ck2=/home/sliu/project_space/datasets/T10/checkpoint-best.pth
# ck3=/home/sliu/project_space/datasets/T15/checkpoint-best.pth
# ck4=/home/sliu/project_space/datasets/T20/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --Decom False  --resume $ck1 --input_size 224 --drop_path 0.1 --data_path $data 
# python main.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --Decom False  --resume $ck2 --input_size 224 --drop_path 0.1 --data_path $data 
# python main.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --Decom False  --resume $ck3 --input_size 224 --drop_path 0.1 --data_path $data 
# python main.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --Decom False  --resume $ck4 --input_size 224 --drop_path 0.1 --data_path $data 


#kernel-size
#ck1=/gpfs/work3/0/prjste21060/projects/datasets/T20_300/checkpoint-best.pth
#python main.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False --data_type=$data_type --resume $ck1 --input_size 224 --drop_path 0.1 --data_path $data

ck1=/gpfs/work3/0/prjste21060/projects/datasets/T20lr4e-3/checkpoint-best.pth
python main.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --data_type $data_type --resume $ck1 --input_size 224 --drop_path 0.1 --data_path $data


# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/3x3/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 3 3 3 3 100 --width_factor 1.0 --bn False --Decom False  --data_type=$data_type --resume $ck1 --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/5x5/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 5 5 5 5 100 --width_factor 1.0 --bn False --Decom False --data_type=$data_type  --resume $ck1 --input_size 224 --drop_path 0.1 --data_path $data

# #data=/home/sliu/project_space/datasets/imagenet-r
# #data=/home/sliu/project_space/datasets/sketch
# #ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/convnext300epoch/checkpoint-best.pth

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/7x7/checkpoint-best.pth
# python main.py --model convnext_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --data_type=$data_type  --Decom True  --resume $ck1 --input_size 224 --drop_path 0.1 --data_path $data


# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/11x11/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 11 11 11 11 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/15x15/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 15 15 15 15 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/21x21/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 21 21 21 21 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/25x25/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 25 25 25 25 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data


# # #kernel-size rep
# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/11x11rep/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 11 11 11 11 5 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data


# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/15x15rep/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 15 15 15 15 5 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/21x21rep/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 21 21 21 21 5 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/25x25rep/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 25 25 25 25 5 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data


# # #kernel-size decomp
# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/15x15decomParalel/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 15 15 15 15 5 --width_factor 1.0 --bn True --Decom True   --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/21x21decomParalel/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 21 21 21 21 5 --width_factor 1.0 --bn True --Decom True   --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/25x25decomParalel/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 25 25 25 25 5 --width_factor 1.0 --bn True --Decom True   --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/31x31decomParalel/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 31 31 31 31 5 --width_factor 1.0 --bn True --Decom True   --resume $ck1 --data_type=$data_type --input_size 224 --drop_path 0.1 --data_path $data


# # #kernel-size decomp sequence
# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/15x15decomSq/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 15 15 15 15 5 --parallel False --width_factor 1.0 --bn True --Decom True --data_type=$data_type   --resume $ck1 --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/21x21decomSq/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 21 21 21 21 5 --parallel False --width_factor 1.0 --bn True --Decom True  --data_type=$data_type  --resume $ck1 --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/25x25decomSq/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 25 25 25 25 5 --parallel False --width_factor 1.0 --bn True --Decom True  --data_type=$data_type  --resume $ck1 --input_size 224 --drop_path 0.1 --data_path $data

# ck1=/gpfs/work3/0/prjste21060/projects/datasets/checkpoints/31x31decomSq/checkpoint-best.pth
# python main.py --model SLaK_tiny --eval True --kernel_size 31 31 31 31 5 --parallel False --width_factor 1.0 --bn True --Decom True --data_type=$data_type   --resume $ck1 --input_size 224 --drop_path 0.1 --data_path $data

source deactivate
