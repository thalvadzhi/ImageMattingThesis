import argparse
from scripts.alpha_matting.predict_alpha_matte import AlphaMattePredictor
from scripts.trimap_generation.predict_trimap import TrimapPredictor
from scripts.saliency.predict_saliency import SaliencyPredictor
from workflow import Workflow
import cv2 as cv
import scipy.misc
alpha_matte_predictor_path = "..\\..\\model_checkpoints\\encoder_decoder.07-val_loss-0.1294-val_sad-187.0877-val_mse-0.0446.hdf5"

parser = argparse.ArgumentParser(
    description='Script for training neural networks.')

# parser.add_argument('--target',
#                     dest='target',
#                     choices=["alpha_matte", "saliency", "trimap"],
#                     action='store',
#                     required=True,
#                     help="What to predict?")

parser.add_argument('--output',
                    dest='output',
                    choices=["alpha", "saliency", "trimap", "alpha_saliency", "alpha_trimap", "saliency_trimap", "all"],
                    action='store',
                    default="all",
                    help="What to save to disk?")

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
def parse_output_argument(output_arg):
    output = {"alpha_matte": False, "saliency": False, "trimap": False}
    if output_arg == "all":
        output["alpha_matte"] = True
        output["saliency"] = True
        output["trimap"] = True
    elif output_arg =="alpha":
        output["alpha_matte"] = True
    elif output_arg == "saliency":
        output["saliency"] = True
    elif output_arg == "trimap":
        output["trimap"] = True
    elif output_arg == "alpha_saliency":
        output["alpha_matte"] = True
        output["saliency"] = True   
    elif output_arg == "alpha_trimap":
        output["alpha_matte"] = True
        output["trimap"] = True  
    elif output_arg == "saliency_trimap":
        output["saliency"] = True
        output["trimap"] = True      

    return output    

args = parser.parse_args()
workflow = Workflow()
what_to_output = parse_output_argument(args.output)

if args.img is None and args.trimap is None and args.saliency is None:
    parser.print_help()
    parser.exit()
else:
    workflow.set_img(args.img)
    workflow.set_sal(args.saliency)
    workflow.set_trimap(args.trimap)
    workflow.workflow(what_to_output)










# if args.target == "alpha_matte":
#     if args.img is None or args.trimap is None:
#         parser.print_help()
#         parser.exit()
#     predictor = AlphaMattePredictor(alpha_matte_predictor_path)
#     predictor.load_model()
#     prediction = predictor.predict_patches(args.img, args.trimap)
#     scipy.misc.imsave('../../output/alpha.png', prediction)
#     # cv.imwrite("alpha.png", prediction)
# elif args.target == "saliency":
#     predictor = SaliencyPredictor("../../model_checkpoints/saliency.34-val_loss-0.9297.hdf5")
#     predictor.load_model()
#     pred = predictor.predict_path(args.img)
#     scipy.misc.imsave('../../output/sal.png', pred)

# elif args.target == "trimap":
#     if args.img is None or args.saliency is None:
#         parser.print_help()
#         parser.exit()
#     predictor = TrimapPredictor()
#     prediction = predictor.predict(args.img, args.saliency)
#     scipy.misc.imsave('../../output/trimap.png', prediction)

