import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

all_img = {}

img = cv.imread("1.jpg", 0)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
all_img["origin"] = img

# 侵蚀
kernel = np.ones((5, 5), np.int8)
erosion = cv.erode(img, kernel, iterations=1)
all_img["erode"] = erosion

# 扩张
dilation = cv.dilate(img, kernel, iterations=1)
all_img["dilation"] = dilation

# 开运算  开放只是**侵蚀然后扩张**的另一个名称。如上文所述，它对于消除噪音很有用
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
all_img["closing"] = closing

# 梯度运算
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
all_img["gradient"] = gradient

# 顶帽
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
all_img["tophat"] = tophat

# 黑帽
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
all_img["blackhat"] = blackhat

for k, v in all_img.items():
    cv.imshow(k, v)
    cv.waitKey(0)

# 结构元素 用于创建内核
# 矩形内核
cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
# 椭圆内核
cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# 十字内核
cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
