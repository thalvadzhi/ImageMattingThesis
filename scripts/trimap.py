import numpy as np
import cv2 as cv

def generate_trimap(alpha):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    fg = cv.erode(fg, kernel, iterations=np.random.randint(3, 10))

    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(5, 20))
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)