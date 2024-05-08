import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = ("poire.jpg")
"""
cv2.IMREAD_COLOR – It specifies to load a color image. Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.
cv2.IMREAD_GRAYSCALE – It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag. 
cv2.IMREAD_UNCHANGED – It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.
"""
img = cv2.imread(filename)
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

laplacian = cv2.Laplacian(imgGrey, cv2.CV_64F, ksize=3)
I = cv2.normalize(laplacian, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

canny = cv2.Canny(imgGrey,threshold1=100.0,threshold2=150.0)

gaussian = cv2.GaussianBlur(imgGrey,(5,5),0)
derivX = cv2.Sobel(gaussian,ddepth=-1,dx=1,dy=0)
derivY = cv2.Sobel(gaussian,ddepth=-1,dx=0,dy=1)
abs_grad_x = cv2.convertScaleAbs(derivX)
abs_grad_y = cv2.convertScaleAbs(derivY)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# show with cv2
cv2.imshow(filename,imgGrey)
cv2.imshow("Laplacian",I)
cv2.imshow("Canny",canny)
cv2.imshow("Sobel",grad)
cv2.waitKey(0)
cv2.destroyAllWindows()

# show with plt
plt.imshow(imgGrey, cmap='gray')
plt.show()