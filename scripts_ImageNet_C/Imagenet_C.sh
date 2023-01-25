#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o ./Imagenet-C.out

source /home/sliu/miniconda3/etc/profile.d/conda.sh
#source activate slak

data=/home/sliu/project_space/datasets/imagenet-c  #改下路径
data_type=imagenet-c

cd ..
####### test teacher models #####################
#swin
python main_robust.py --model swin --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#vit-s
python main_robust.py --model vit --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#cswin
ck1=/cks/Teachers/cswin_tiny_224.pth
python main_robust.py --model cswin --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#convnext-t 300ep
ck1=/cks/Teachers/checkpoint-best.pth
python main_robust.py --model convnext --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#slak-T 
ck1=/cks/Teachers/SLaK_tiny_checkpoint.pth
python main_robust.py --model SLaK_tiny --eval True --kernel_size 51 49 47 13 5 --width_factor 1.3 --bn True --Decom True  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data


##############test student models #############

#student: convnext-T
#original convnext-T 120ep
ck1=/cks/Student/convnext_120ep/checkpoint-best.pth
python main_robust.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#distill from swin
ck1=/cks/bnTrue_TeacherSwin_STConvNext_NKD/checkpoint-best.pth
python main_robust.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data

#distill from cswin
ck1=/cks/bnTrue_TeacherCSwin_STConvNext_NKD/checkpoint-best.pth
python main_robust.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data

#distill from convnext
ck1=/cks/bnTrue_ConvNeXtTeacher_STconvnextv1_NKD/checkpoint-best.pth
python main_robust.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data

#distill from SLakT
ck1=/cks/bnTrue_SLaKTTeacher_STConvNext_NKD/checkpoint-best.pth
python main_robust.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data


#student: resnet-50

#distill from vit
ck1=/cks/bnTrue_VitTeacher_STRN50_NKD/checkpoint-best.pth
python main.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#distill from swin
ck1=/cks/bnTrue_SwinTeacher_STRN50_NKD/checkpoint-best.pth
python main.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#distill from cswin
ck1=/cks/bnTrue_TeachersCSwin_STRN50_NKD/checkpoint-best.pth
python main.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data

#distill from convNeXt
ck1=/cks/bnTrue_ConvNeXtTeacher_STRN50_NKD/checkpoint-best.pth
python main.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data

#distill from SLaK-T
ck1=/cks/bnTrue_SLaKTTeachers_STRN50_NKD/checkpoint-best.pth
python main.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data



source deactivate
