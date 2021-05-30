# These codes can used to generate fake background for the original image during training
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import linecache
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import random
import torch
def color_transfer(Is,Ms): # [h,w,c]

    colorfunction=random.choice([color,bright,color_enhancement])
    
    It = colorfunction(Is, Ms)
    return It
def color(Is,Ms):
    txtname = './dataset/real_dataset_for_finetune/train.txt'
    count = len(open(txtname, 'r').readlines())
    I2num = random.randrange(1, count, 1)
    data = linecache.getline(txtname, I2num)
    name = data.split('\n')[0]

    It = cv2.imread('./dataset/real_dataset_for_finetune/real/' + name)
    
    
    # BGR->LAB
    Is = Is.float()
    LabIs = Is.permute(2,0,1)/255
    LabIs = BGR2RGB(LabIs)
    LabIs = rgb2lab(LabIs)

    It = It.astype(np.float32)/255
    It = torch.from_numpy(It)
    It = It.permute(2,0,1)
    It = BGR2RGB(It)
    It = rgb2lab(It)
    
    # mean、std
    Is_means = torch.zeros([3,1])
    It_means = torch.zeros([3,1])
    Is_stdevs = torch.zeros([3,1])
    It_stdevs = torch.zeros([3,1])
    LabIs = LabIs / 255.
    It = It / 255.
    
    for i in range(3):
        Is_means[i,:] += torch.mean(LabIs[i, :, :])
        It_means[i,:] += torch.mean(It[i, :, :])
        Is_stdevs[i,:]  += torch.std(LabIs[i, :, :])
        It_stdevs[i,:]  += torch.std(It[i, :, :])
    
    thresh = [a / b for a, b in zip(It_stdevs, Is_stdevs)]
    thresh = It_stdevs / Is_stdevs
    thresh = torch.stack([thresh[0,:], thresh[2,:], thresh[1,:]], dim = 0)
    LabIt = torch.zeros_like(It)
    
    for i in range(3):        
        LabIt[i,:,:] = thresh[i,:]*(LabIs[i,:,:]-Is_means[i,:]) + It_means[i,:]
    
#     # [0-255]
    LabIt = (LabIt * 255.)
    LabIt *= (LabIt > 0)
    LabIt = (LabIt * (LabIt <= 255) + 255 * (LabIt > 255))


    It = LabIt
    It = lab2rgb(It)

    It = RGB2BGR(It)
    It = It.permute(1,2,0)
    It = It*255
    
    # show
    It = Is * (Ms / 255) + It * (1 - Ms / 255)
    return It
def bright(Is,Ms):
    It = torch.zeros_like(Is)
    a = random.uniform(-100, 200)
    It = Is + a
    It = torch.clamp(It,0,255)
    It = Is * (Ms / 255) + It * (1 - Ms / 255)
    return It
def color_enhancement(Is,Ms):
    It = MSRCR(
        Is,
        [15, 80, 200],
        5.0, #G
        25.0, #b
        125.0, #alpha
        46.0 #beta
    )
    It = Is * (Ms / 255) + It * (1 - Ms / 255)
    return It

#########################################################################
# color enhancement code, refer from https://blog.csdn.net/weixin_38285131/article/details/88097771
def singleScaleRetinex(img,sigma):
    retinex = torch.log10(img) - torch.log10(torch.from_numpy(cv2.GaussianBlur(img.numpy(), (0, 0), sigma)))
    return retinex

def multiScaleRetinex(img, sigma_list):

    retinex = torch.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex


def colorRestoration(img, alpha, beta):

    img_sum = torch.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (torch.log10(alpha * img) - torch.log10(img_sum))

    return color_restoration

  


def MSRCR(img, sigma_list, G, b, alpha, beta):
    
    img = img.float()+1.0


    img_retinex = multiScaleRetinex(img, sigma_list)
    
    img_color = colorRestoration(img, alpha, beta)    
    img_msrcr = G * (img_retinex * img_color + b)

    
    img_msrcr *= (img_msrcr > 0)
    img_msrcr = (img_msrcr * (img_msrcr <= 255) + 255 * (img_msrcr > 255))

    return img_msrcr


#########################################################################
# color rgb2lab code , refer from https://blog.csdn.net/Ly_MinSheng/article/details/110231726
def BGR2RGB(img): 
    return torch.stack([img[2,:,:], img[1,:,:], img[0,:,:]], dim = 0)
def RGB2BGR(img): 
    return torch.stack([img[2,:,:], img[1,:,:], img[0,:,:]], dim = 0)

def F(X): 
    FX = 7.787*X + 0.137931
    index = X > 0.008856
    FX[index] = torch.pow(X[index],1.0/3.0)
    return FX
def anti_F(X):
    tFX = (X - 0.137931) / 7.787
    index = X > 0.206893
    tFX[index] = torch.pow(X[index],3)
    return tFX



def gamma(r):
    r2 = r / 12.92
    index = r > 0.04045 
    r2[index] = torch.pow((r[index] + 0.055) / 1.055, 2.4)
    return r2
def anti_g(r):
    r2 = r*12.92
    index = r > 0.0031308072830676845
    r2[index] = torch.pow(r[index], 1.0/2.4)*1.055 - 0.055
    return r2



def rgb2lab(img): 
    r = img[0,:,:]
    g = img[1,:,:]
    b = img[2,:,:]

    r = gamma(r)
    g = gamma(g)
    b = gamma(b)

    X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
    Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
    Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470
    X = X / 0.964221
    Z = Z / 0.825211

    F_X = F(X)
    F_Y = F(Y)
    F_Z = F(Z)


    L = 116*F_Y - 16.0
    a = 500*(F_X-F_Y) 
    b = 200*(F_Y-F_Z) 

    return torch.stack([L, a, b], dim = 0)

def lab2rgb(Lab):
    fY = (Lab[0,:,:] + 16.0) / 116.0
    fX = Lab[1,:,:] / 500.0 + fY
    fZ = fY - Lab[2,:,:] / 200.0

    x = anti_F(fX)
    y = anti_F(fY)
    z = anti_F(fZ)
    x = x * 0.964221
    z = z * 0.825211
    #
    r = 3.13405134*x - 1.61702771*y - 0.49065221*z
    g = -0.97876273*x + 1.91614223*y + 0.03344963*z
    b = 0.07194258*x - 0.22897118*y + 1.40521831*z
    # 
    r = anti_g(r)
    g = anti_g(g)
    b = anti_g(b)
    return torch.stack([r, g, b], dim = 0).clamp(0.0,1.0)
if __name__ == '__main__':
    I1 = cv2.imread("dataset/real_dataset_for_finetune/real/1803151818-00000003.png")
    M1 = cv2.imread("dataset/real_dataset_for_finetune/alpha/1803151818-00000003.png")
    I1 = torch.from_numpy(I1)
    M1 = torch.from_numpy(M1)
    It = color_transfer(I1, M1)
    It = np.array(It, dtype='uint8')
    cv2.imwrite("./change.png",It)

