from scripts.alpha_matting.predict_alpha_matte import AlphaMattePredictor
from scripts.trimap_generation.predict_trimap import TrimapPredictor
from scripts.saliency.predict_saliency import SaliencyPredictor
import cv2 as cv
import scipy
alpha_matte_predictor_path = "../../model_checkpoints/encoder_decoder.07-val_loss-0.1294-val_sad-187.0877-val_mse-0.0446.hdf5"
# alpha_matte_predictor_path = "../../model_checkpoints/refined.hdf5"

saliency_predictor_path = "../../model_checkpoints/saliency.34-val_loss-0.9297.hdf5"
SALIENCY_PATH = "../../output/saliency.png"
TRIMAP_PATH = "../../output/trimap.png"
ALPHA_MATTE_PATH = "../../output/alpha.png"

class Workflow:
    def __init__(self, img=None, sal=None, trimap=None):
        self.img = img
        self.sal = sal
        self.trimap = trimap
    
    def set_img(self, img):
        self.img = cv.imread(img)
    
    def set_sal(self, sal):
        self.sal = cv.imread(sal) / 255
        print(self.sal.shape)
    
    def set_trimap(self, trimap):
        self.trimap = cv.imread(trimap)

    def _only_image_workflow(self, output):
        print("Predicting saliency...")
        saliency_predictor = SaliencyPredictor(saliency_predictor_path)
        saliency_predictor.load_model()
        saliency = saliency_predictor.predict_img(self.img)
        print("Done with saliency!")
        if output["saliency"] is True:
            print("Saving saliency...")
            self.save_img(saliency, SALIENCY_PATH)

        self._img_saliency_workflow(output, self.img, saliency)

    
    def _img_saliency_workflow(self, output, img, saliency):
        if output["trimap"] is False and output["alpha_matte"] is False:
            # we've done our work
            return
        saliency_shape = saliency.shape
        if len(saliency_shape) == 3 and saliency_shape[2] == 3:
            saliency = saliency[:, :, 0] 
        print("Predicting trimap...")
        trimap_predictor = TrimapPredictor()
        trimap = trimap_predictor.predict_img(img, saliency)
        print("Done with trimap!")
        if output["trimap"] is True:
            print("Saving trimap...")
            self.save_img(trimap, TRIMAP_PATH)
        
        self._img_trimap_workflow(output, img, trimap)
    
    def _img_trimap_workflow(self, output, img, trimap):

        trimap_shape = trimap.shape
        if len(trimap_shape) == 3 and trimap_shape[2] == 3:
            trimap = trimap[:, :, 0]
        trimap = trimap.reshape(trimap_shape[0], trimap_shape[1], 1)
    
        if output["alpha_matte"] is False:
            return
        print("Predicting alpha matte...")
        alpha_matte_predictor = AlphaMattePredictor(alpha_matte_predictor_path)
        alpha_matte_predictor.load_model()
        alpha_matte = alpha_matte_predictor.predict_patches_img(img, trimap)
        print("Done with alpha matte!")
        print("Saving alpha matte.")
        self.save_img(alpha_matte, ALPHA_MATTE_PATH)

    
    def workflow(self, output):
        if (self.img is not None) and (self.sal is None) and (self.trimap is None):
            self._only_image_workflow(output)
        elif (self.img is not None) and (self.sal is not None) and (self.trimap is None):
            self._img_saliency_workflow(output, self.img, self.sal)
        elif (self.img is not None) and (self.sal is None) and (self.trimap is not None):
            self._img_trimap_workflow(output, self.img, self.trimap)


    def save_img(self, img, path):
        scipy.misc.imsave(path, img)
