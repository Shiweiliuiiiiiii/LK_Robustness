data=/workspace/ssd1/shiwei/data/Imagenet-robustness/Imagenet-c #改下路径
data_type=imagenet-c

cd ..
####### test teacher models #####################
#swin
python main_robust.py --model swin --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data

#vit-s
python main_robust.py --model vit --eval True --kernel_size 7 7 7 7 100 --width_factor 1.0 --bn True --Decom False  --resume $ck1 --data_type $data_type --input_size 224 --drop_path 0.1 --data_path $data



source deactivate
