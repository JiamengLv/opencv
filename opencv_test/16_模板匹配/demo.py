import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('1.jpg', 0)
img2 = img.copy()
template = cv.imread('2.jpg', 0)
w,  h = template.shape[::-1]

# 列表中所有的6种比较方法

methods = ['cv.TM_CCOEFF',  'cv.TM_CCOEFF_NORMED',  'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED',  'cv.TM_SQDIFF',  'cv.TM_SQDIFF_NORMED']

for meth in methods:

    img = img2.copy()
    method = eval(meth)          # 将字符串str当成有效的表达式来求值并返回计算结果,返回索引

    print(method)

    # 应用模板匹配
    res = cv.matchTemplate(img, template, method)

    # minMaxLoc（）返回最小值，最大值的位置     -----》 单模板匹配
    min_val,  max_val,  min_loc,  max_loc = cv.minMaxLoc(res)

    print(min_loc)

    # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
    if method in [cv.TM_SQDIFF,  cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc


    bottom_right = (top_left[0] + w,  top_left[1] + h)

    # 画矩形  图片 框的大小  坐标
    cv.rectangle(img, top_left,  bottom_right,  255,  2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'),  plt.xticks([]),  plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'),  plt.xticks([]),  plt.yticks([])
    plt.suptitle(meth)
    plt.show()


# 模板匹配建议使用归一化后的图片