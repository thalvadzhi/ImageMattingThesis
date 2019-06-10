from trimap import generate_trimap
import cv2 as cv
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import numpy as np
base = "D:\\Image Matting Dataset\\adobe dataset\\Combined_Dataset\\Training_set\\all_alphas\\"
img_name = "1-1255621189mTnS.jpg"
img = cv.imread(base + img_name)
# imshow(img)
# plt.show()

alpha = img
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
fg = np.equal(alpha, 255).astype(np.float32)
fg = cv.erode(fg, kernel, iterations=10)

# imshow(fg)
# plt.show()
unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
# imshow(unknown)
unknown = cv.dilate(unknown, kernel, iterations=20)
# imshow(unknown)
# plt.show()
trimap = fg * 255 + (unknown - fg) * 128
imshow(trimap.astype(np.uint8))
plt.show()