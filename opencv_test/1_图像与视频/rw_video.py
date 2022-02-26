import numpy as np
import cv2 as cv

# 读取摄像头，我笔记本这里有一个外置的，所以为1，使用本机自带改为0
cap = cv.VideoCapture(1)

# 定义编解码器并创建VideoWriter对象
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

if not cap.isOpened():
    print('camera is closed')
    exit()
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive (stream end?) . Exiting ...')
        break
    # 写操作
    out.write(frame)

    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()

