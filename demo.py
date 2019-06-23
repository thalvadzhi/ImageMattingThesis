import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2 as cv
import numpy as np
from workflow import Workflow
WINDOW_NAME = "demo"
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

def init():
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

capture = cv.VideoCapture(0)
init()
while True:
    ret, frame = capture.read()
    frame = cv.flip(frame, 1)
    cv.imshow(WINDOW_NAME, frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


what_to_output = {"alpha_matte": True, "saliency": True, "trimap": True}
workflow = Workflow()
workflow.set_img(frame)
workflow.workflow(what_to_output)

alpha_matte = workflow.alpha_matte
am_shape = alpha_matte.shape
bg = cv.GaussianBlur(frame.astype(np.uint8),(155,155),0)
alpha_matte = alpha_matte.reshape(am_shape[0], am_shape[1], 1)
bg_hawaii = cv.imread("../../images/hawaii2.jpg")

bg_crop = bg_hawaii[0:alpha_matte.shape[0], 0:alpha_matte.shape[1]]

cv.imshow("blurred", ((frame * alpha_matte) + (1-alpha_matte) * bg).astype(np.uint8))
cv.imshow("hawaii", ((frame * alpha_matte) + (1-alpha_matte) * bg_crop).astype(np.uint8))
print("we're here")

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()