import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('1.jpg')


def show_img(image):
    cv.imshow('img', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
# 缩放
res = cv.resize(img, None, fx=2,  fy=2,  interpolation=cv.INTER_CUBIC)
show_img(res)

height, width = img.shape[:2]
res = cv.resize(img, (2*width,  2*height),  interpolation=cv.INTER_CUBIC)
show_img(res)

# 平移
height, width = img.shape[:2]
M = np.float32([[1, 0, 100], [0, 1, 50]])
res = cv.warpAffine(img, M, (height, width))
show_img(res)

# 旋转
height, width = img.shape[:2]
M = cv.getRotationMatrix2D(((height-1)/2.0, (width-1)/2.0), 90, 1)
res = cv.warpAffine(img, M, (height, width))
show_img(res)

# 仿射变换
img = cv.imread('1.jpg')
rows, cols, ch = img.shape
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(img, M, (cols, rows))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()

# 透视变换
img = cv.imread('1.jpg')
rows, cols, ch = img.shape
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(img, M, (300, 300))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()

