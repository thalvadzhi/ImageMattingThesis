import sys
import keras
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D
sys.path.insert(0, "../")
from scripts.alpha_matting.load_encoder_decoder import build_encoder_decoder_from_vgg
from scripts.alpha_matting.data_generator import DataGenerator
from scripts.alpha_matting.losses import overall_loss_wrapper, sad_wrapper, mse_wrapper, alpha_loss_wrapper
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.models import load_model
from scripts.unpooling import Unpooling
from scripts.alpha_matting.load_refinement import build_refinement
from alt_model_checkpoint import AltModelCheckpoint
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from scripts.alpha_matting.run_encoder_decoder_arch import TrainEncoderDecoder
import os
import cv2 as cv
from patches import get_all_patches, combine_patches, get_crops_by_squares, combine_predictions, patches_of_culture
from trimap import get_final_output
import numpy as np

class AlphaMattePredictor:
    def __init__(self, path_to_predictor):
        self.path_to_predictor = path_to_predictor

    def load_model(self, ):
        self.model = load_model(self.path_to_predictor, custom_objects={"Unpooling": Unpooling()})
    
    def predict_patches_img(self, img, trimap):
        patches, shape = get_all_patches(img, trimap)

        predictions = []
        for patch in patches:
            predictions.append(self.model.predict(patch.reshape(1, 320, 320, 4)))
        
        n = len(predictions)
        combined_predicton = combine_patches(np.array(predictions).reshape(n, 320, 320, 1), (shape[0], shape[1], 1))

        return get_final_output(combined_predicton.reshape(shape[0], shape[1]), trimap[:, :, 0] / 255)
    
    def predict_patches_path(self, path_to_img, path_to_trimap):
        img = cv.imread(path_to_img)
        trimap = cv.imread(path_to_trimap)
        return self.predict_patches_img(img, trimap)

    def predict_patches_intelligent_img(self, img, trimap):
        squares = patches_of_culture(img, trimap[:, :, 0])
        patches = get_crops_by_squares(img, trimap[:, :, 0:1], squares)

        predictions = []
        for patch in patches:
            predictions.append(self.model.predict((patch.reshape(1, 320, 320, 4))).reshape(320, 320, 1))
        output = combine_predictions(predictions, squares, img.shape)
        print(output.shape)
        return get_final_output(output.reshape(img.shape[0], img.shape[1]), trimap[:, :, 0] / 255)

    
    def predict_patches_intelligent_path(self, path_to_img, path_to_trimap):
        img = cv.imread(path_to_img)
        trimap = cv.imread(path_to_trimap)
        return self.predict_patches_intelligent_img(img, trimap)

    def predict_resized_img(self, img, trimap):
        original_shape = img.shape
        img_resized = cv.resize(img, (320, 320))
        trimap_resized = cv.resize(trimap, (320, 320))

        input_for_model = np.zeros((320, 320, 4))
        input_for_model[:, :, 0:3] = img_resized
        input_for_model[:, :, 3] = trimap_resized

        prediction = self.model.predict(input_for_model.reshape(1, 320, 320, 4))
        
        return cv.resize(prediction[0], (original_shape[1], original_shape[0]))

    def predict_resized_path(self, path_to_img, path_to_trimap):
        img = cv.imread(path_to_img)
        trimap = cv.imread(path_to_trimap)
        return self.predict_resized_img(img, trimap)




