data=/workspace/ssd1/shiwei/data/Imagenet-robustness/Imagenet-c #改下路径
data_type=imagenet-c

####### test teacher models #####################

#cswin
ck1=/workspace/ssd1/shiwei/LK_distill/Table_1_CKS/RN50/checkpoint-best.pth
python main_robust.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data


source deactivate
