import sys
sys.path.insert(0, "../../")
from scripts.saliency.refine_saliency import *
from scripts.trimap import generate_trimap
import cv2 as cv
from skimage.segmentation import slic, mark_boundaries


class TrimapPredictor:
    def __init__(self, n_superpixels=600):
        self.n_superpixels = n_superpixels
    
    def predict_img(self, img, saliency):
        
        segments = slic(img, n_segments = self.n_superpixels, compactness=50)
        medians = get_median_superpixel(saliency, segments)
        sal, classes = classify_superpixels_based_on_median_of_saliency(saliency, segments, medians)
        hists = get_color_hist_for_each_superpixel(img, segments)
        dist = get_bhat_d_for_each_pair_superpixels_fast(hists)
        new_fg, new_bg = clusterize_superpixels_hist(dist, classes, hists)
        saliency_new = color_saliency(new_fg, new_bg, segments, saliency)
        try:
            ret, otsu = cv.threshold((saliency_new * 255).astype(np.uint8)[:, :, 0],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        except IndexError:
            ret, otsu = cv.threshold((saliency_new * 255).astype(np.uint8),0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        
        return generate_trimap(otsu)
    def predict_path(self, img_path, saliency_path):
        img = cv.imread(img_path)
        saliency = cv.imread(saliency_path) / 255
        return self.predict_img(img, saliency)