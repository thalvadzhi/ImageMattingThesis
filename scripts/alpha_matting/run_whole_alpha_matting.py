import sys
import keras
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D
sys.path.insert(0, "../../scripts/")
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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

class TrainWholeAlphaMatting(TrainEncoderDecoder):
    def __init__(self, batch_size=16, n_gpus=2, lr=0.00001, model_name="refined.hdf5"):
        super().__init__(batch_size, n_gpus, lr)
        self.model_name = model_name
    
    def _get_callbacks(self, model):
        tensor_board = keras.callbacks.TensorBoard(log_dir=TrainWholeAlphaMatting.PATH_LOGS, histogram_freq=0, write_graph=True, write_images=True)
        model_name = TrainWholeAlphaMatting.PATH_MODEL_CHECKPOINTS + 'whole_alpha_matting.{epoch:02d}-val_loss-{val_loss:.4f}-val_sad-{val_sad:.4f}-val_mse-{val_mse:.4f}.hdf5'
        model_checkpoint = AltModelCheckpoint(filepath=model_name, alternate_model=model,  monitor='val_loss', verbose=1, save_best_only=True)

        early_stop = EarlyStopping('val_loss', patience=6, verbose=1)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=3, verbose=1)

        return [tensor_board, model_checkpoint, early_stop, reduce_lr]

    def load_model(self):
        refinement = load_model(TrainWholeAlphaMatting.PATH_MODEL_CHECKPOINTS + self.model_name, custom_objects={"Unpooling": Unpooling()})
       
        
        self.train_gen, self.test_gen = self._get_data_generators(self.batch_size)

        self.callbacks = self._get_callbacks(refinement)
        
        self.multi_gpu_model = multi_gpu_model(refinement, gpus=self.n_gpus)

        inputs = refinement.input

        optimizer = Adam(lr=self.lr)
        self.multi_gpu_model.compile(optimizer=optimizer, loss=alpha_loss_wrapper(inputs), metrics=[sad_wrapper(inputs), mse_wrapper(inputs)])