import numpy as np
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import cv2 as cv
import os

BASE = "D:\\Image Matting Dataset\\images with transparency\\"
all_pngs = [name for name in os.listdir(BASE) if name.endswith(".png") and (not "alpha" in name)]
for png_name in all_pngs:
    name_no_extension = os.path.splitext(png_name)[0]
    image = cv.imread(BASE + name_no_extension + ".png", cv.IMREAD_UNCHANGED)
    print(png_name)
    cv.imwrite(name_no_extension + "_alpha.png", image[:, :, 3])
