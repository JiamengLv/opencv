import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv.imread('2.jpg')
s = 100
img = cv.resize(img, (s, s))
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("rgb", rgb)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.medianBlur(gray, 5)
_, thresh = cv.threshold(blur, 10, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# Search for contours and select the biggest one
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

h, w = img.shape[:2]
mask = np.ones((h, w), np.uint8)
cv.drawContours(mask, contours, -1, 0, -1)
res = cv.bitwise_and(img, img, mask=mask)
x, y, w, h = 2, 70, 30, 40
cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    if x <= 0 or y <= 0:
        print(x, y, w, h)
    elif x + w > s or y + h > s:
        print(x, y, w, h)
    else:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv.imshow("img", img)
cv.imshow("res", res)
cv.waitKey(0)
cv.destroyAllWindows()
