pip install timm==0.4.5 tensorboardX six

# base
python main.py --model convnext_base --eval true --resume /home2/pytorch-broad-models/ConvNext/convnext_base_22k_1k_224.pth --input_size 224 --drop_path 0.2 --data_path /home2/pytorch-broad-models/imagenet/raw/
# small
python main.py --model convnext_small --eval true --resume /home2/pytorch-broad-models/ConvNext/convnext_small_22k_1k_224.pth --input_size 224 --drop_path 0.2 --data_path /home2/pytorch-broad-models/imagenet/raw/
# large
python main.py --model convnext_large --eval true --resume /home2/pytorch-broad-models/ConvNext/convnext_large_22k_1k_224.pth --input_size 224 --drop_path 0.2 --data_path /home2/pytorch-broad-models/imagenet/raw/
