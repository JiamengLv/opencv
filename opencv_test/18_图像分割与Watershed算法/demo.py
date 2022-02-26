import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("2.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# 现在我们需要去除图像中的任何白点噪声。为此，我们可以使用形态学扩张。
# 要去除对象中的任何小孔，我们可以使用形态学侵蚀。
# 我们需要提取我们可确定为硬币的区域。侵蚀会去除边界像素。
# 因此，无论剩余多少，我们都可以肯定它是硬币。

# 噪声去除
kernel = np.ones((1, 1), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
# 确定背景区域
sure_bg = cv.dilate(thresh, kernel, iterations=3)
# 寻找前景区域
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# 找到未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# 类别标记
ret, markers = cv.connectedComponents(sure_fg)
# 为所有的标记加1，保证背景是0而不是1
markers = markers+1
# 现在让所有的未知区域为0
markers[unknown == 255] = 0

# 使用plt.imshow();plt.show()显示每个阶段的图
print(1)

