import cv2
import numpy as np
import matplotlib as plt
#%%
img1 = cv2.imread('C:\\Users\\User\\Desktop\\NTUST\\image_processing\\Hw2\\HW2 readme\\sky.jpg' , 1)
#%%
hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

#%%
cv2.imshow('My Image1', img_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
get_sky = np.zeros(img1.shape,dtype = np.uint8)
for channel in range(3):
    sky_pixel_row , sky_pixel_col = np.where(img_mask > 30)
    for position in range(len(sky_pixel_row)):
        get_sky[sky_pixel_row[position] , sky_pixel_col[position]] = img1[sky_pixel_row[position] , sky_pixel_col[position]]
#%%
cv2.imshow('My Image1', get_sky)
cv2.waitKey(0)
cv2.destroyAllWindows()