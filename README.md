# FGFormer

To run the model FGFormer-B/4, use this command inside the project folder:

torchrun --nnodes=1 --nproc_per_node=N train.py --model FGFormer-B/4 --data-path /path/to/imagenet/train

N is the number GPUs used for training.
