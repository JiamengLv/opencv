# 基础绘图，绘制基本图形
import numpy as np
import cv2 as cv

img = np.zeros((512, 512, 3))
# 线 -- 参数：图片， x，y，颜色，线宽
cv.line(img, (0, 0), (512, 512), (255, 0, 0), 5)

# 矩形
cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# 圆  -1表示填充
cv.circle(img, (447, 63), 63, (0, 0, 255), -1)

# 椭圆  图片，中心点坐标，轴坐标，角度，开始角，结束角，颜色，填充/线宽
cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)

# 多边形
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]],  np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img, [pts], True, (0, 255, 255))

# 文字信息
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

cv.imshow('img', img)

cv.waitKey(0)
cv.destroyAllWindows()
