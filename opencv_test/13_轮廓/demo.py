import numpy as np
import cv2 as cv

im = cv.imread('1.jpg')

imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# 为了准确行一般建议化成二值的灰度图
ret, thresh = cv.threshold(imgray, 127, 255, 0)

# mode 轮廓检索模式
# RETR_TREE : 默认检测所有的轮廓，并按照嵌套的结构保存所有的轮廓
# method 轮廓的逼近方法
# cv.CHAIN_APPROX_NONE
# cv.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# 轮廓点 嵌套结构

# 传入绘制图像、轮廓、轮廓索引(可以选择显示轮廓)、颜色模式、线条厚度
# 注意需要copy，要不原图会变
draw_image = im.copy()
a = cv.drawContours(draw_image, contours, -1, (0, 255, 0), 1)
# a = cv.drawContours(draw_image, contours, 0, (0, 255, 0), 1)
cv.imshow("a", a)
cv.waitKey(0)


#  计算轮廓的一些 特征
# 特征矩  cv.moments(contours[0])
# 轮廓面积  cv.contourArea(contours[0])
# 轮廓周长  cv.arcLength(contours[0],True)

# 轮廓近似
# cnt = contours[0]
# epsilon = 0.1 * cv.arcLength(cnt, True)        # 一般按照轮阔的周长进行设置
# approx = cv.approxPolyDP(cnt, epsilon, True)

# 轮廓凸包  cv.convexHull(contours[0])

# 边界矩形
# 1 直角矩形
# x,y,w,h = cv.boundingRect(cnt)        # 使用bounding找出轮廓的外接矩形
# cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# 2 旋转矩形
# rect = cv.minAreaRect(cnt)
# box = cv.boxPoints(rect)
# box = np.int0(box)
# cv.drawContours(img,[box],0,(0,0,255),2)

# 最小闭合圈
# (x,y),radius = cv.minEnclosingCircle(cnt)
# center = (int(x),int(y))
# radius = int(radius)
# cv.circle(img,center,radius,(0,255,0),2)

# 拟合一个椭圆
# ellipse = cv.fitEllipse(cnt)
# cv.ellipse(img,ellipse,(0,255,0),2)

# 拟合直线
# rows,cols = img.shape[:2]
# [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

# 轮廓属性
# 凸性缺陷
# hull = cv.convexHull(cnt,returnPoints = False)
# defects = cv.convexityDefects(cnt,hull)

# 点多边形测试
# dist = cv.pointPolygonTest(cnt,(50,50),True)

# 形状匹配
# cv.matchShapes(cnt1,cnt2,1,0.0)

# 轮廓层级

# 轮廓检索模式
# RETR_LIST  正常检索
# RETR_EXTERNAL  检索外部轮廓
# RETR_CCOMP
# RETR_TREE
