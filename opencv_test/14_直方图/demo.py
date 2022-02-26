import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('1.jpg')

# image 原图像 图像格式为 unit8，float32，当传入函数时 应用中括号[]
# channel 如果图像是灰度图 值为[0], 如果是彩色图像 传入的参数 是 [0][1][2] 分别对应着BGR通道
# mask 如果都要统计的话就置为0，是一个全为 255 的numpy，遮盖住不想统计的区域
# histSize bin的数目
# ranges 像素值的范围

# 直方图的计算
hist = cv.calcHist([img], [0], None, [256], [0, 256])

plt.plot(hist)
plt.show()

# 创建一个掩码 mask
mask = np.zeros(img.shape[:2])
mask[100:300,100:400] = 255
# cv.imshow("mask",mask)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 将原始图 与 mask 进行与操作，展现想要的区域
# masked_img = cv.bitwise_and(img,img,mask=mask)
# cv.imshow("mask_img",masked_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# numpy
# 1
# hist, bins = np.histogram(img.ravel(), 256, [0, 256])
# 2
hist = np.bincount(img.ravel(), minlength=256)
plt.plot(hist)
plt.show()

# matplotlib
# plt.hist(img.ravel(), 256, [0, 256])
color = ('b', 'g', 'r')
plt.figure()
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    # plt.xlim([0, 256])
plt.show()
