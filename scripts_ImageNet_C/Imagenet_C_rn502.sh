data=/ssd1/shiwei/data/Imagenet-robustness/Imagenet-c
data_type=imagenet-c


#student: resnet-50


#distill from cswin
ck1=/ssd1/shiwei/LK_distill/Table_1_CKS/bnTrue_TeachersCSwin_STRN50_NKD/checkpoint-best.pth
python main.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data

#distill from convNeXt
ck1=/ssd1/shiwei/LK_distill/Table_1_CKS/bnTrue_ConvNeXtTeacher_STRN50_NKD/checkpoint-best.pth
python main.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data

#distill from SLaK-T
ck1=/ssd1/shiwei/LK_distill/Table_1_CKS/bnTrue_SLaKTTeachers_STRN50_NKD/checkpoint-best.pth
python main.py --model resnet50 --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn False --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data



source deactivate
