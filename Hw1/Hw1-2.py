# 引入所需要的函式
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
origin_img = cv2.imread('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\HW1\\camaraMan.png',0)
print(origin_img.shape)
#%% # mean_filter

#new_height = new_width = (W — F + 1) / S 
# 有padding : floor[(W - F + 2 * P + 1) / S] + 1

def mean_filter(img,kernel_size):
    #定義kernel 此處使用 mean filter
    kernel = np.ones([kernel_size , kernel_size]) / (kernel_size * kernel_size)
    print(kernel)
    # 先 padding 因為 convolution後size會縮小
    convolve_img = np.zeros([img.shape[0] + kernel_size - 1, img.shape[1] + kernel_size - 1])
    convolve_img[:img.shape[0],:img.shape[1]] = img # 將原圖放入要進行捲積的array
    print(convolve_img.shape)
    res_img = np.zeros([img.shape[0],img.shape[1]])
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            conv_arr =  convolve_img[row : row + kernel_size , col : col + kernel_size]
            res_img[row][col] = np.sum(conv_arr * kernel)
    print(res_img.shape)
    return res_img
    
#%%
mean_img = mean_filter(origin_img,5)
# USM = 0.8 * (a-b) + a
UM_img = 0.8 * (origin_img - mean_img ) + origin_img
#%% #畫圖
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(3, 1, 1)
imgplot = plt.imshow(origin_img,'gray')
ax.set_title('origin')
ax.axis('off')
ax = fig.add_subplot(3, 1, 2)
imgplot = plt.imshow(mean_img,'gray')
ax.set_title('after mean_filter')
ax.axis('off')
ax = fig.add_subplot(3, 1, 3)
imgplot = plt.imshow(UM_img,'gray')
ax.set_title('Unsharp Masking')
ax.axis('off')

