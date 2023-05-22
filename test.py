import cv2
import numpy as np

src = cv2.imread("src.png")
dst = cv2.imread("dst.jpg")

tl = [915,254]
bl = [160,520]
tr = [1692,288]
br = [1088,567]

tl2 = [72,61]
bl2 = [72,181]
tr2 = [130,61]
br2= [130,181]

pts1 = np.float32([tl,bl,tr,br])
pts2 = np.float32([tl2,bl2,tr2,br2])

pts3 = np.float32([[672,851]])
pts4 = np.float32([[390,926]])

M = cv2.getPerspectiveTransform(pts1,pts2)

pts3o=cv2.perspectiveTransform(pts3[None, :, :], M)
pts4o=cv2.perspectiveTransform(pts4[None, :, :], M)

x = int(pts3o[0][0][0])
y = int(pts3o[0][0][1])
p = (x,y)

x1 = int(pts4o[0][0][0])
y1 = int(pts4o[0][0][1])
p1 = (x1,y1)

cv2.circle(dst,p,5,(0,0,255),-1)
cv2.circle(dst,p1,5,(0,0,255),-1)

cv2.imshow("sada", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()