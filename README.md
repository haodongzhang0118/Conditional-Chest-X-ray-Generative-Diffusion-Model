# Conditional Chest X-ray Generative Diffusion Model with FGFormer

In this project, we use FGFormer-B/4 as our model for training and sampling. Here is the completely instruction for the project.

# Result Presentation
![samples](https://github.com/haodongzhang0118/Conditional-Chest-X-ray-Generative-Diffusion-Model/assets/86388854/576cc953-1a06-4891-9ec5-da3afe42fbaf)

The denoise processing has been included in the github repo which is called **generation_gif.gif**

# Demo:

The project.ipynb is the simple demo that using our model generates an image based on random labels (0 - 13)

We recommend that using a GPU to run it, othervise it may take 12 mins.

Before using the project.ipynb, download checkpoints from the google drive with the link: https://drive.google.com/drive/folders/1Prr-fasZ2LHesbOABDAFgiin2PmS3M2q?usp=sharing
There are several checkpoints in it. We recommend to use 0090000.pt which is the latest with 90000 training steps.

Putting them into the same folder to avoid errors.

project.html is the html version of the finished running project.ipynb.


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
