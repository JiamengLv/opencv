import numpy as np
import cv2 as cv

# 加法运算
x = np.uint8([250])
y = np.uint8([10])
# OpenCV加法是饱和运算，而Numpy加法是模运算。
# 饱和运算，就是当运算结果大于一个上限或小于一个下限时，结果就等于上限或是下限。
# 模运算就是取余
print(cv.add(x, y))  # 250+10 = 260 => 255
print(x+y)  # 250+10 = 260 % 256 = 4

# 融合，注意需要两张图片大小相同
img1 = cv.imread('4.jpg')
img2 = cv.imread('5.jpg')
dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 位运算
# 加载两张图片
img1 = cv.imread('3.jpg')
img2 = cv.imread('5.jpg')
# 我想把logo放在左上角，所以我创建了ROI
rows, cols, channels = img2.shape
roi = img1[0:rows,  0:cols]
# 现在创建logo的掩码，并同时创建其相反掩码
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray,  10,  100,  cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# 现在将ROI中logo的区域涂黑
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
# 仅从logo图像中提取logo区域
img2_fg = cv.bitwise_and(img2, img2, mask=mask)
# 将logo放入ROI并修改主图像
dst = cv.add(img1_bg, img2_fg)
img1[0:rows,  0:cols] = dst
cv.imshow('res', img1)
cv.waitKey(0)
cv.destroyAllWindows()
