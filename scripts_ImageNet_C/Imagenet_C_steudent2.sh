data=/ssd1/shiwei/data/Imagenet-robustness/Imagenet-c #改下路径
data_type=imagenet-c

####### test teacher models #####################

#cswin
ck1=/ssd1/shiwei/LK_distill/Table_1_CKS/Teachers/cswin_tiny_224.pth
python main_robust.py --model cswin --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#convnext-t 300ep
ck1=/ssd1/shiwei/LK_distill/Table_1_CKS/Teachers/checkpoint-best.pth
python main_robust.py --model convnext --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#slak-T 
ck1=/ssd1/shiwei/LK_distill/Table_1_CKS/Teachers/SLaK_tiny_checkpoint.pth
python main_robust.py --model SLaK_tiny --eval True --kernel_size 51 49 47 13 5 --width_factor 1.3 --bn True --Decom True  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data


source deactivate
