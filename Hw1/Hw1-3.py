import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
origin_img = cv2.imread('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\HW1\\ntust_gray.jpg',0)
print(origin_img.shape)
#%%

#new_height = new_width = (W — F + 1) / S 
# 有padding : floor[(W - F + 2 * P + 1) / S] + 1

def median_filter(img,kernel_size):
    # 先 padding 因為 convolution後size會縮小
    convolve_img = np.zeros([img.shape[0] + kernel_size - 1, img.shape[1] + kernel_size - 1])
    # 因kernel 為 3*3
    convolve_img[:img.shape[0],:img.shape[1]] = img
    print(convolve_img.shape)
    res_img = np.zeros([img.shape[0],img.shape[1]])
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            res_img[row][col] = np.median(convolve_img[row:row + kernel_size,col:col + kernel_size])
            
    return res_img
    
#%%
# 加噪點
noise_num = int(origin_img.shape[0] * origin_img.shape[1] * 0.15) # 總pixel的15%
noise_img = origin_img.copy()

for loop in range(noise_num):
    row = int(np.random.uniform(0, origin_img.shape[0])) # random choose
    col = int(np.random.uniform(0, origin_img.shape[1])) 
    noise_img[row,col] = 255 # 將選到的位置設為255
filter_img = noise_img.copy()
res_img = median_filter(filter_img,3)
print(res_img.shape)

#%% #畫圖
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(3, 1, 1)
imgplot = plt.imshow(origin_img,'gray')
ax.set_title('origin')
ax.axis('off')
ax = fig.add_subplot(3, 1, 2)
imgplot = plt.imshow(noise_img,'gray')
ax.set_title('noise_15%')
ax.axis('off')
ax = fig.add_subplot(3, 1, 3)
imgplot = plt.imshow(res_img,'gray')
ax.set_title('median_filter')
ax.axis('off')
#%%
write_path = 'C:\\Users\\boy09\\OneDrive\\Documents\\image processing ppt\\HW1 readme\\output\\3'
outputList = ['origin.jpg','salt.jpg','res.jpg']
out_img = [origin_img,noise_img,res_img]
for i ,j in zip(outputList,out_img):
    outpath = os.path.join(write_path,i)
    cv2.imwrite(outpath,j)
