import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 
#%%
origin_img = cv2.imread('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\HW1\\ntust_gray.jpg',0)
print(origin_img.shape)
#%%
origin_img = origin_img / 255
#%%
# 0 for horizontal derivative
# 1 for vertical derivative
#vertical= np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) Gy
#horizontal = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) Gx
def sobel_filter(img , axis):
    if axis == 0:
        kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        res = cv2.filter2D(img,-1,kernel)
    elif axis == 1:
        kernel =  np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        res = cv2.filter2D(img,-1,kernel)
    return res
#%%
gx = sobel_filter(origin_img,0) # horizontal
gy = sobel_filter(origin_img,1) # vertical
relief = gx + 0.5
np.absolute(gx) # get abs
np.absolute(gy) 
# set threshold = 0.25
_ ,thresgx = cv2.threshold(gx,0.25,1,cv2.THRESH_BINARY)
_ ,thresgy = cv2.threshold(gy,0.25,1,cv2.THRESH_BINARY)
# result mix gx and gy
sobel = cv2.bitwise_or(thresgx,thresgy)
#%% #畫圖
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(3, 1, 1)
imgplot = plt.imshow(origin_img,'gray')
ax.set_title('origin')
ax.axis('off')
ax = fig.add_subplot(3, 1, 2)
imgplot = plt.imshow(relief,'gray')
ax.set_title('gx + 0.5')
ax.axis('off')
ax = fig.add_subplot(3, 1, 3)
imgplot = plt.imshow(sobel,'gray')
ax.set_title('after sobel')
ax.axis('off')
#%%
write_path = 'C:\\Users\\boy09\\OneDrive\\Documents\\image processing ppt\\HW1 readme\\output\\1'
outputList = ['origin.jpg','relief.jpg','sobel.jpg']
out_img = [origin_img*255 , relief*255, sobel*255]
for i ,j in zip(outputList,out_img):
    outpath = os.path.join(write_path,i)
    cv2.imwrite(outpath,j)