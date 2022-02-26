# 读写单张图片
import numpy as np
import cv2 as cv

# 即使图像路径错误，它也不会引发任何错误，但是 print img 会给出 None

# cv.IMREAD_COLOR  加载彩色图像。任何图像的透明度都会被忽视。它是默认标志
# cv.IMREAD_GRAYSCALE  以灰度模式加载图像
# cv.IMREAD_UNCHANGED  加载图像，包括alpha通道

img = cv.imread('1.jpg', 0)
cv.imshow('image', img)

# 一个键盘绑定函数。其参数是以毫秒为单位的时间。该函数等待任何键盘事件指定的毫秒。
# 如果您在这段时间内按下任何键，程序将继续运行。如果**0**被传递，它将无限期地等
# 待一次敲击键。它也可以设置为检测特定的按键，例如，如果按下键 a 等，我们将在下面讨论。

# 如果使用的是64位计算机，则必须 k = cv.waitKey(0) 按如下所示修改行： k = cv.waitKey(0) & 0xFF

k = cv.waitKey(0)
if k == ord('s'):  # 等待关键字，保存和退出
    cv.imwrite('2.png', img)
    cv.destroyAllWindows()

# OpenCV加载的彩色图像处于BGR模式。但是Matplotlib以RGB模式显示。


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi5.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])    # 隐藏 x 轴和 y 轴上的刻度值
plt.show()