import cv2
import numpy as np
import matplotlib.pyplot as plt
#%% 讀檔

img_sky = cv2.imread('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\sky.jpg')
img_sky_mask = cv2.imread('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\sky_mask.jpg')
img1 = cv2.imread('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\bird.jpg')
img1 = cv2.resize(img1, (800, 500), interpolation=cv2.INTER_CUBIC)
img2 = cv2.imread('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\bird2.jpg')
img2 = cv2.resize(img2, (800, 500), interpolation=cv2.INTER_CUBIC)

#%%
get_sky = np.zeros(img_sky.shape,dtype = np.uint8) #宣告全為0的array ,將藍天的值放入
for channel in range(3):
    sky_pixel_row , sky_pixel_col = np.where(img_sky_mask[:,:,channel] > 50) #對 BGR channel 分別執行 得到哪些值大於50 (表不為黑)
    get_sky[sky_pixel_row , sky_pixel_col] = img_sky[sky_pixel_row , sky_pixel_col] #將藍天pixel的值 對入get_sky對應的位置
#%% 轉換色彩空間
sky_HSV = cv2.cvtColor(get_sky, cv2.COLOR_BGR2HSV)
sky_YCC = cv2.cvtColor(get_sky, cv2.COLOR_BGR2YCR_CB)
cv2.imshow('BGR' , get_sky)
cv2.imshow('HSV' , sky_HSV)
cv2.imshow('YCC' , sky_YCC)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%% get BGR , HSV ,  YCC mean and std
# get BGR , HSV ,  YCC mean and std
# --------BGR---------
B = get_sky[:,:,0]
G = get_sky[:,:,1]
R = get_sky[:,:,2]
print('B,G,R mean {} , {} , {} ' .format(np.mean( B[B> 50] ) ,np.mean( G[G > 50] ) ,np.mean( R[R > 50] )))
print('B,G,R std {} , {} , {} ' .format(np.std( B[B> 50] ) ,np.std( G[G > 50] ) ,np.std( R[R > 50] )))
# --------HSV---------
H = sky_HSV[:,:,0]
S = sky_HSV[:,:,1]
V = sky_HSV[:,:,2]
print('H,S,V mean {} , {} , {} ' .format(np.mean( H[H > 50] ) ,np.mean( S[S > 50] ) ,np.mean( V[V > 50] )))
print('H,S,V std {} , {} , {} ' .format(np.std( H[H > 50] ) ,np.std( S[S > 50] ) ,np.std( V[V > 50] )))
# --------YCrCb---------
Y  = sky_YCC[:,:,0]
Cr = sky_YCC[:,:,1]
Cb = sky_YCC[:,:,2]
print('Y,Cr,Cb mean {} , {} , {} ' .format(np.mean( Y[Y > 50] ) ,np.mean( Cr[Cr > 50] ) ,np.mean( Cb[Cb > 50] )))
print('Y,Cr,Cb std {} , {} , {} ' .format(np.std( Y[Y > 50] ) ,np.std( Cr[Cr > 50] ) ,np.std( Cb[Cb > 50] )))

#%%
# 將個別的mean , std存入array 做後續運算
BGR_mean = np.array([np.mean( B[B> 50]) , np.mean( G[G > 50]) , np.mean( R[R > 50] ) ]) 
BGR_std = np.array([np.std( B[B> 50]) , np.std( G[G > 50]) , np.std( R[R > 50] ) ]) 
HSV_mean = np.array([ np.mean( H[H > 50] ), np.mean( S[S > 50] ) , np.mean( V[V > 50] ) ])
HSV_std = np.array([ np.std( H[H > 50] ), np.std( S[S > 50] ) , np.std( V[V > 50] ) ])
YCrCb_mean = np.array([np.mean( Y[Y> 50]) , np.mean( Cr[Cr > 50]) , np.mean( Cb[Cb > 50] ) ]) 
YCrCb_std = np.array([np.std( Y[Y> 50]) , np.std( Cr[Cr > 50]) , np.std( Cb[Cb > 50] ) ])
#%% img1
def get_sky_region(img , mean , std):
    binary_img = np.zeros(img.shape , dtype = np.uint8) #二值化的圖
    res_img = np.zeros(img.shape , dtype = np.uint8) #藍天的圖
    colorSpace_upper = mean + 2 * std # mean加上兩個std
    colorSpace_lower = mean - 2 * std # mean減|掉兩個std
    rule_pass = np.zeros(img.shape , dtype = bool) #最後用此標籤判定是否為藍天 ， 三個channel都需要介於upper lower分別對應的值
    for channel in range(3):
        sky_row , sky_col = np.where((img[:,:,channel] >= colorSpace_lower[channel]) & (img[:,:,channel] <= colorSpace_upper[channel]))
        # 找到pixel值有介於upper lower之間的位置 ， 分別對每個channel運算
        rule_pass[sky_row , sky_col ,channel] = True
    index = np.sum(rule_pass , axis = 2)
    poix, poiy = np.where(index == 3) # 要三個channel都符合在範圍內才算是藍天
    res_img[poix, poiy] = img[poix, poiy] # 得到該圖的值
    binary_img[poix, poiy] = 255 
    return res_img , binary_img
#%%
# img1
img1_BGR ,img1_BGR_mask = get_sky_region(img1, BGR_mean , BGR_std)
img1_HSV ,img1_HSV_mask = get_sky_region(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV) , HSV_mean , HSV_std)
img1_YCC ,img1_YCC_mask = get_sky_region(cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB) , YCrCb_mean , YCrCb_std)
#img2
img2_BGR ,img2_BGR_mask = get_sky_region(img2, BGR_mean , BGR_std)
img2_HSV ,img2_HSV_mask = get_sky_region(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV) , HSV_mean , HSV_std)
img2_YCC ,img2_YCC_mask = get_sky_region(cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB) , YCrCb_mean , YCrCb_std)

#%%
# img1 
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img1_BGR.jpg', img1_BGR)
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img1_HSV.jpg', img1_HSV)
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img1_YCC.jpg', img1_YCC)
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img1_BGR_mask.jpg', img1_BGR_mask)
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img1_HSV_mask.jpg', img1_HSV_mask)
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img1_YCC_mask.jpg', img1_YCC_mask)
# img2
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img2_BGR.jpg', img2_BGR)
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img2_HSV.jpg', img2_HSV)
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img2_YCC.jpg', img2_YCC)
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img2_BGR_mask.jpg', img2_BGR_mask)
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img2_HSV_mask.jpg', img2_HSV_mask)
cv2.imwrite('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\result\\img2_YCC_mask.jpg', img2_YCC_mask)