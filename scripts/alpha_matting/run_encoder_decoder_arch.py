#!/usr/bin/env python

import sys
import keras
from alt_model_checkpoint import AltModelCheckpoint
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D
sys.path.insert(0, "../../scripts/")
from scripts.alpha_matting.load_encoder_decoder import build_encoder_decoder_from_vgg
from scripts.alpha_matting.data_generator import DataGenerator
from scripts.alpha_matting.losses import overall_loss_wrapper, sad_wrapper, mse_wrapper
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

class TrainEncoderDecoder:
    PATIENCE = 30
    BACKGROUNDS_PER_FG_TRAIN = 100
    BACKGROUNDS_PER_FG_TEST = 20
    PATH_MODEL_CHECKPOINTS = "../../model_checkpoints/"
    PATH_LOGS = "../../logs"
    HOME = "../../"

    def __init__(self, batch_size=16, n_gpus=2, lr=0.00001):
        self.batch_size = batch_size
        self.n_gpus = n_gpus
        self.lr = lr
    
    def load_model(self):
        inputs, model = build_encoder_decoder_from_vgg()
       
        self.train_gen, self.test_gen = self._get_data_generators(self.batch_size)

        self.callbacks = self._get_callbacks(model)
        

        self.multi_gpu_model = multi_gpu_model(model, gpus=self.n_gpus)

        optimizer = Adam(lr=self.lr)
        self.multi_gpu_model.compile(optimizer=optimizer, loss=overall_loss_wrapper(inputs), metrics=[sad_wrapper(inputs), mse_wrapper(inputs)])

    def _get_callbacks(self, model):
        tensor_board = keras.callbacks.TensorBoard(log_dir=TrainEncoderDecoder.PATH_LOGS, histogram_freq=0, write_graph=True, write_images=True)
        model_name = TrainEncoderDecoder.PATH_MODEL_CHECKPOINTS + 'encoder_decoder.{epoch:02d}-val_loss-{val_loss:.4f}-val_sad-{val_sad:.4f}-val_mse-{val_mse:.4f}.hdf5'
        model_checkpoint = AltModelCheckpoint(filepath=model_name,alternate_model=model,  monitor='val_loss', verbose=1, save_best_only=True)

        # model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True)
        early_stop = EarlyStopping('val_loss', patience=TrainEncoderDecoder.PATIENCE, verbose=1)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=TrainEncoderDecoder.PATIENCE // 4, verbose=1)

        return [tensor_board, model_checkpoint, early_stop, reduce_lr]

    def _get_data_generators(self, batch_size):
        base_train = TrainEncoderDecoder.HOME + "train_ds/"
        base_test  = TrainEncoderDecoder.HOME + "test_ds/"

        fg_train = base_train + "foregrounds/"
        bg_train = base_train + "backgrounds/"
        a_train = base_train + "alphas/"

        fg_names_train = base_train + "training_fg_names.txt"
        bg_names_train = base_train + "training_bg_names.txt"

        fg_test = base_test + "foregrounds/"
        bg_test = base_test + "backgrounds/"
        a_test = base_test + "alphas/"

        fg_names_test = base_test + "test_fg_names.txt"
        bg_names_test = base_test + "test_bg_names.txt"

        train_gen = DataGenerator(fg_names_train, bg_names_train, TrainEncoderDecoder.BACKGROUNDS_PER_FG_TRAIN, fg_train, bg_train, a_train, batch_size, True)
        test_gen = DataGenerator(fg_names_test, bg_names_test, TrainEncoderDecoder.BACKGROUNDS_PER_FG_TEST, fg_test, bg_test, a_test, batch_size, True)
        return train_gen, test_gen

    def train(self, epochs=200):
        return self.multi_gpu_model.fit_generator(
            self.train_gen,
            epochs=epochs,
            verbose=1,
            callbacks=self.callbacks,
            validation_data=self.test_gen,
            shuffle=True,
            workers=3,
            use_multiprocessing=True)
