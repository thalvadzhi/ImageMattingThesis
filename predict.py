import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
from scripts.alpha_matting.predict_alpha_matte import AlphaMattePredictor
from scripts.trimap_generation.predict_trimap import TrimapPredictor
from scripts.saliency.predict_saliency import SaliencyPredictor
from workflow import Workflow
import cv2 as cv
import scipy.misc

parser = argparse.ArgumentParser(
    description='Script for predicting with neural networks.')

parser.add_argument('--output',
                    dest='output',
                    choices=["alpha", "saliency", "trimap", "alpha_saliency", "alpha_trimap", "saliency_trimap", "all"],
                    action='store',
                    default="all",
                    help="What to save to disk?")

parser.add_argument('--img',
                    dest='img',
                    required=True,
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
