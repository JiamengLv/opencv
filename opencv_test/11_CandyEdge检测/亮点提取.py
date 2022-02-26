import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
# 读入灰度图像
img = cv.imread('2.jpg', 0)

# 阈值127分割图像
ret, th = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# 找出非零点
x,y =np.where(th!=0)
z = list(zip(x,y))
print(len(z))

plt.imshow(th)
plt.show()


