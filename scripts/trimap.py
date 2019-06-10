import numpy as np
import cv2 as cv

def generate_trimap(alpha):
    kernel_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    fg = np.equal(alpha, 255).astype(np.float32)
    fg = cv.erode(fg, kernel_erode, iterations=2)#np.random.randint(3, 10))

    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel_dilate, iterations=2)#np.random.randint(5, 20))
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)