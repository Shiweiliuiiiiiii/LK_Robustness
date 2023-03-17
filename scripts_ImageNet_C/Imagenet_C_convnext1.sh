data=/workspace/ssd1/shiwei/data/Imagenet-robustness/Imagenet-c
data_type=imagenet-c



export PATH='/usr/local/cuda-11.4/bin:$PATH'
export LD_LIBRARY_PATH='/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH'
export CUDA_PATH='/usr/local/cuda-11.4'

ck1=/workspace/ssd1/shiwei/LK_distill/slak_Convnext_t_300_epoch/checkpoint-best.pth
CUDA_VISIBLE_DEVICES=7 python main_robust.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 \
--bn True --Decom False  --resume /workspace/ssd1/shiwei/LK_distill/slak_Convnext_t_300_epoch/checkpoint-best.pth --data_type imagenet-c --input_size 224 --drop_path 0.1 --data_path /workspace/ssd1/shiwei/data/Imagenet-robustness/Imagenet-c  \
--output_dir /workspace/ssd1/shiwei/LK_distill/Table_1_CKS/Student/convnext_300ep/


source deactivate





#original convnext-T 300ep




##ck1=/workspace/ssd1/shiwei/LK_distill/Table_1_CKS/Student/convnext_120ep/checkpoint-best.pth
#CUDA_VISIBLE_DEVICES=7 python main_robust.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 \
#--bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data  \
#--output_dir /workspace/ssd1/shiwei/LK_distill/Table_1_CKS/Student/convnext_120ep/



