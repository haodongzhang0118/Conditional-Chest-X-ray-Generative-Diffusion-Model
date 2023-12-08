# Conditional Chest X-ray Generative Diffusion Model with FGFormer

In this project, we use FGFormer-B/4 as our model for training and sampling

Here is the completely instruction for this project:

# Dataset

The dataset we used is MIMIC-CXR-JPG v2.0.0, which is about 400 ~ 500 GB. It is impossible to upload it into the github. However, it can be downloaded from the website https://physionet.org/content/mimic-cxr-jpg/2.0.0/.

The original structure of dataset is:
root/

----set/

--------subject/

------------study/

----------------image.png + report.txt

After downloading the dataset, using python scripts in PreProcess/LabelTask/ process data, guiding by the csv files in CSVFiles Label/. This will make the dataset folder become:
Dataset Train (80015 images)
root/
----label1

--------image1.png

--------image2.png

----label2

.

----label14


Dataset Test (20000 images)
root/

----label1

--------image1.png

--------image2.png

----label2

.

----label14


# Training

After finishing pre-processing the dataset, using the following commend to train the model:

torchrun --nnodes=1 --nproc_per_node=N train.py --model FGFormer-B/4 --data-path /path/to/data/train --num-classes 14

N is the number GPUs used for training.

# Sampling

After finishing training the model, using this commend to smaple an image generated from Gaussian Noise by trained model's ckpt

python sample.py --model FGFormer-B/4 --image-size 256 --ckpt /path/to/model.pt --num-sampling-steps 1000 --num-classes 14

# Creating a npz file for future FID and F1 calculation

Using this commend to creat a npz file

torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model FGFormer-B/4 --image-size 256 --ckpt /path/to/model.pt --num-sampling-steps 1000 --num-classes 14
