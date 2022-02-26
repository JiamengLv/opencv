import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# opencv中主要就是cv2.dft() 和 cv2.idft(), 输入图像先要转换成np.float32 格式
# 得到的结果中频率为0的部分会在左上角，通常要转换到中心位置，可以通过shift变换来实现
#  cv2.dft() 返回的结果是双通道的（实部，虚部），通常还需要转换成图像格式才能显示（0,255）
# 频域上的低通 高通操作


img = cv.imread("./1.jpg",0)

img_float32 = np.float32(img)

dft = cv.dft(img_float32, flags=cv.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.shift(dft)
# 得到灰度图能表示形式
magnitue_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

rows,cols = dft_shift.shape
crow,ccol = int(rows/2,cols/2)        # 中心位置

# 低通滤波
mask = np.zeros((rows,cols,2),np.unit8)
mask[crow-30:crow+30,ccol-30:ccol+30] = 1

# IDFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])





