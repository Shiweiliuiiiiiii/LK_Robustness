data=/workspace/ssd1/shiwei/data/Imagenet-robustness/Imagenet-c
data_type=imagenet-c


##############test student models #############

#student: convnext-T
#original convnext-T 120ep
ck1=/ssd1/shiwei/LK_distill/Table_1_CKS/Student/convnext_120ep/checkpoint-best.pthcd
python main_robust.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data
python
#distill from swin
ck1=/ssd1/shiwei/LK_distill/Table_1_CKS/bnTrue_TeacherSwin_STConvNext_NKD/checkpoint-best.pth
python main_robust.py --model SLaK_tiny --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --input_size 224 --data_type $data_type --drop_path 0.1 --data_path $data




source deactivate
