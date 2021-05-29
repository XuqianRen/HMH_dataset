import os
import random
import cv2
import numpy as np
data_dir="./dataset/real_dataset_for_finetune"
data_list=os.path.join(data_dir,'train.txt')
bg_path = os.path.join(data_dir,'bg')
fg_path = os.path.join(data_dir,'fg')
img_path = os.path.join(data_dir,'real')
alpha_path = os.path.join(data_dir,'alpha')

os.makedirs(bg_path,exist_ok=True)
os.makedirs(fg_path,exist_ok=True)
with open(data_list) as datalist:
      for lines in datalist.readlines():
            data =lines.split("\n")
            image_name=data[0]            
            img=cv2.imread(os.path.join(img_path,image_name))
            alpha=cv2.imread(os.path.join(alpha_path,image_name))

            bg_img = cv2.inpaint(img, alpha[:,:,0].astype('uint8'), 15, cv2.INPAINT_TELEA)
            fg_img = img*(alpha/255)
            
            cv2.imwrite(os.path.join(bg_path,image_name), bg_img)
            cv2.imwrite(os.path.join(fg_path,image_name), fg_img)

            print(image_name)
            