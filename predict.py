import argparse
from scripts.alpha_matting.predict_alpha_matte import AlphaMattePredictor
from scripts.trimap_generation.predict_trimap import TrimapPredictor
import cv2 as cv
import scipy.misc
alpha_matte_predictor_path = "..\\..\\model_checkpoints\\encoder_decoder.07-val_loss-0.1294-val_sad-187.0877-val_mse-0.0446.hdf5"

parser = argparse.ArgumentParser(
    description='Script for training neural networks.')

parser.add_argument('--target',
                    dest='target',
                    choices=["alpha_matte", "saliency", "trimap"],
                    action='store',
                    required=True,
                    help="What to predict?")

parser.add_argument('--img',
                    dest='img',
                    action='store',
                    help="The input image for alpha matte, saliency, trimap generation")

parser.add_argument('--trimap',
                    dest='trimap',
                    action='store',
                    help="The input trimap for alpha matte") 

parser.add_argument('--saliency',
                    dest='saliency',
                    action='store',
                    help="The input saliency map for trimap generation")                

args = parser.parse_args()
if args.target == "alpha_matte":
    if args.img is None or args.trimap is None:
        parser.print_help()
        parser.exit()
    predictor = AlphaMattePredictor(alpha_matte_predictor_path)
    predictor.load_model()
    prediction = predictor.predict_patches(args.img, args.trimap)
    scipy.misc.imsave('../../output/alpha.png', prediction)
    # cv.imwrite("alpha.png", prediction)
elif args.target == "saliency":
    pass
elif args.target == "trimap":
    if args.img is None or args.saliency is None:
        parser.print_help()
        parser.exit()
    predictor = TrimapPredictor()
    prediction = predictor.predict(args.img, args.saliency)
    scipy.misc.imsave('../../output/trimap.png', prediction)

