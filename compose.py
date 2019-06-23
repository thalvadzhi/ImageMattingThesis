from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(
    description='Script for composing image with new background.')

parser.add_argument('--img',
                    dest='img',
                    required=True,
                    action='store',
                    help="The input image for composition.")

parser.add_argument('--alpha_matte',
                    dest='alpha_matte',
                    required=True,
                    action='store',
                    help="The alpha matte to use for the composition.")

parser.add_argument('--background',
                    dest='background',
                    required=True,
                    action='store',
                    help="The background to use for the composition.")

parser.add_argument("--show", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Show the composed image.")

parser.add_argument("--save", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Save the composed image.")

args = parser.parse_args()
img = cv.resize(cv.imread(args.img), (1280, 720))
alpha_matte = cv.imread(args.alpha_matte) / 255


if img.shape[0] != alpha_matte.shape[0] or img.shape[1] != alpha_matte.shape[1]:
    raise Exception("Input image shape does not match alpha matte shape - {} != {}".format(img.shape, alpha_matte.shape))

if args.background == "blur":
    bg = cv.GaussianBlur(img.astype(np.uint8),(155,155),0)
else:
    bg = cv.imread(args.background)

if bg.shape[0] > alpha_matte.shape[0] and bg.shape[1] > alpha_matte.shape[1]:
    print(bg.shape, alpha_matte.shape)
    bg = bg[0:alpha_matte.shape[0], 0:alpha_matte.shape[1]]
else:
    bg = cv.resize(bg, (alpha_matte.shape[1], alpha_matte.shape[0]))

composition = ((img * alpha_matte) + (1 - alpha_matte) * bg).astype(np.uint8)

if args.save is True:
    cv.imwrite("output/composed.png", composition)

if args.show is True:
    cv.imshow("Composition", composition)
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break