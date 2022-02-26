# 鼠标事件
import numpy as np
import cv2 as cv
events = [i for i in dir(cv) if 'EVENT' in i]
print(events)


def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)


drawing = False  # 如果按下鼠标，则为真
mode = True  # 如果为真，绘制矩形。按 m 键可以切换到曲线
ix, iy = -1, -1


# 鼠标回调函数
def draw_circle2(event, x, y, flags, param):
    global ix, iy, drawing, mode
    if event == cv.EVENT_LBUTTONDOWN:  # 左键按下
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:  # 移动鼠标
        if drawing:
            if mode:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:  # 左键抬起
        drawing = False
        if mode:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONDBLCLK:  # 双击
        mode = False if mode else True


img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle2)
while 1:
    cv.imshow('image', img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()
