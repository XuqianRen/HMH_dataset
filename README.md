# A Generative Adversarial Framework for Optimizing Image Matting and Harmonization Simultaneously

## Introduction

In this repository, we release our dataset and the codes for the dataset generating pipeline.

We generate our training dataset based on a public dataset: https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets. 

Our generated HMH dataset can be downloaded from https://1drv.ms/u/s!Aqnc8W8pU0scakyZpbg3j-mIHw4?e=V4ZdDY

The dataset contains the original real images and their corresponding alpha images, and it also contains the foreground and background images we extracted from the real images.

The dataset generating pipeline code can be found in generate.py. We also provide the color change transfer codes in color_transfer.py,  which purpose to change the background of the images.