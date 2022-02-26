# 图像卷积
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
kernel = np.ones((5, 5), np.float32)/25
kernel = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])
dst = cv.filter2D(img, -1, kernel)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]),  plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]),  plt.yticks([])
plt.show()
