import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('2.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 平均模糊 - 仅获取内核区域下所有像素的平均值，并替换中心元素
blur = cv.blur(img, (3,3))
# 高斯模糊 - 指定内核的宽度和高度，该宽度和高度应为正数和奇数
gaussian_blur = cv.GaussianBlur(img, (3,3), 0)
# 中位模糊 - 提取内核区域下所有像素的中值，并将中心元素替换为该中值。这对于消除图像中的椒盐噪声非常有效。
median_blur = cv.medianBlur(img, 5)
# 双边滤波 - 在去除噪声的同时保持边缘清晰锐利非常有效。但是，与其他过滤器相比，该操作速度较慢
bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)

plt.subplot(231), plt.imshow(img), plt.title('Original')
plt.xticks([]),  plt.yticks([])

plt.subplot(232), plt.imshow(blur), plt.title('AverageBlurred')
plt.xticks([]),  plt.yticks([])

plt.subplot(233), plt.imshow(gaussian_blur), plt.title('GaussianBlurred')
plt.xticks([]),  plt.yticks([])

plt.subplot(234), plt.imshow(median_blur), plt.title('MedianBlurred')
plt.xticks([]),  plt.yticks([])

plt.subplot(235), plt.imshow(bilateral_blur), plt.title('BilateralBlurred')
plt.xticks([]),  plt.yticks([])

plt.show()

# 另一种显示方式
res = np.hstack((blur,gaussian_blur,median_blur,bilateral_blur))
cv.imshow("13",res)
cv.waitKey(0)
cv.destroyAllWindows()


