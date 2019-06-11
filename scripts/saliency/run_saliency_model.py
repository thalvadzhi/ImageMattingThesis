import sys
import keras
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D
sys.path.insert(0, "../")

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, ZeroPadding2D, Activation
from keras.optimizers import Adam
import numpy as np
# from unpooling import Unpooling
from matplotlib.pyplot import imshow, figure
# from model import build_encoder_decoder, build_refinement
import cv2 as cv
import math
import tensorflow as tf
from scipy import misc
from scripts.saliency.load_saliency_model import build_saliency_model
from scripts.saliency.saliency_data_generator import get_data_generators
from keras.losses import binary_crossentropy

def bin_croos_entr_wrapper(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred, from_logits=True)

losses = {
    "conv4_side1": "binary_crossentropy",
    "upsampling_concat2": "binary_crossentropy",
    "upsampling_concat3": "binary_crossentropy",
    "upsampling_concat4": "binary_crossentropy",
    "upsampling_side5": "binary_crossentropy",
    "upsampling_side_pool5": "binary_crossentropy",
    "conv_fuse": "binary_crossentropy"
}

loss_weights = {
    "conv4_side1": 1,
    "upsampling_concat2": 1,
    "upsampling_concat3": 1,
    "upsampling_concat4": 1,
    "upsampling_side5": 1,
    "upsampling_side_pool5": 1.0,
    "conv_fuse": 1
}


class TrainSaliencyModel:
    PATIENCE = 30
    PATH_MODEL_CHECKPOINTS = "../../model_checkpoints/"
    PATH_LOGS = "../../logs"
    HOME = "../../"

    def __init__(self, batch_size=16, lr=0.0001):
        self.batch_size = batch_size
        self.lr = lr

    def _get_callbacks(self):
        tensor_board = keras.callbacks.TensorBoard(
            log_dir=TrainSaliencyModel.PATH_LOGS,
            histogram_freq=0,
            write_graph=True,
            write_images=True)
        model_name = TrainSaliencyModel.PATH_MODEL_CHECKPOINTS + 'saliency.{epoch:02d}-val_loss-{val_loss:.4f}.hdf5'
        model_checkpoint = ModelCheckpoint(filepath=model_name,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True)

        # model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True)
        early_stop = EarlyStopping('val_loss',
                                   patience=TrainSaliencyModel.PATIENCE,
                                   verbose=1)
        reduce_lr = ReduceLROnPlateau('val_loss',
                                      factor=0.1,
                                      patience=TrainSaliencyModel.PATIENCE //
                                      4,
                                      verbose=1)

        return [tensor_board, model_checkpoint, early_stop, reduce_lr]

    def _get_data_generators(self, batch_size=16):
        return get_data_generators(batch_size=batch_size)

    def load_model(self):
        self.model, self.outputs = build_saliency_model()
        self.train_gen, self.test_gen = self._get_data_generators()
        self.callbacks = self._get_callbacks()
        optimizer = Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer,
                           loss=losses,
                           loss_weights=loss_weights)

    def train(self, epochs=200):
        return self.model.fit_generator(self.train_gen,
                                        epochs=epochs,
                                        verbose=1,
                                        callbacks=self.callbacks,
                                        validation_data=self.test_gen,
                                        shuffle=True,
                                        )

    def train_debug(self, epochs=1):
        return self.model.fit_generator(self.train_gen,
                                        epochs=epochs,
                                        steps_per_epoch=1,
                                        verbose=1,
                                        callbacks=self.callbacks,
                                        validation_data=self.test_gen,
                                        validation_steps=1,
                                        shuffle=True,
                                        )

