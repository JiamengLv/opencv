# cv.ADAPTIVE_THRESH_MEAN_C::阈值是邻近区域的平均值减去常数**C**。
# cv.ADAPTIVE_THRESH_GAUSSIAN_C:阈值是邻域值的高斯加权总和减去常数**C**。
import cv2 as cv

# 自适应阈值
img = cv.imread('1.jpg', 0)
thresh_types = [cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, cv.THRESH_TRUNC, cv.THRESH_TOZERO,
                cv.THRESH_TOZERO_INV]

for thresh_type in thresh_types:
    ret, thresh = cv.threshold(img, 127, 255, thresh_type)
    cv.imshow("%s" % str(thresh_type), thresh)
    cv.waitKey(0)
cv.destroyAllWindows()
