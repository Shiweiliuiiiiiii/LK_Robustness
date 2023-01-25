data=/workspace/ssd1/shiwei/data/Imagenet-robustness/Imagenet-c
data_type=imagenet-c


#student: resnet-50

#distill from vit
ck1=/workspace/ssd1/shiwei/LK_distill/Table_1_CKS/bnTrue_VitTeacher_STRN50_NKD/checkpoint-best.pth
python main_robust.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#distill from swin
ck1=/workspace/ssd1/shiwei/LK_distill/Table_1_CKS/bnTrue_SwinTeacher_STRN50_NKD/checkpoint-best.pth
python main_robust.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data




source deactivate
