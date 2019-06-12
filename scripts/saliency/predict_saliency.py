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
from patches import get_all_patches, combine_patches
from trimap import get_final_output
import numpy as np

class SaliencyPredictor:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        self.model = load_model(self.model_path)
    
    def predict_img(self, img):
        original_shape = img.shape
        img_resized = cv.resize(img, (320, 448)).reshape(1, 448, 320, 3)
        prediction = self.model.predict(img_resized / 255)
        z_fuse = prediction[6]
        z2 = prediction[1]
        z3 = prediction[2]
        z4 = prediction[3]
        avg = (z_fuse + z2 + z3 + z4) / 4
        avg = z_fuse.reshape(448, 320, 1)
        image_back_to_normal_size = cv.resize(avg, (original_shape[1], original_shape[0]))
        return image_back_to_normal_size

    def predict_path(self, img_path):
        img = cv.imread(img_path)
        return self.predict_img(img)