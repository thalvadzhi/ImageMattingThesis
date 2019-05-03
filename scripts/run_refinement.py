import sys
import keras
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D
sys.path.insert(0, "../scripts/")
from load_encoder_decoder import build_encoder_decoder_from_vgg
from data_generator import DataGenerator
from losses import overall_loss_wrapper, sad_wrapper, mse_wrapper, alpha_loss_wrapper
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.models import load_model
from unpooling import Unpooling
from load_refinement import build_refinement
from alt_model_checkpoint import AltModelCheckpoint
from keras.utils import multi_gpu_model
from keras.optimizers import Adam

model = load_model("../model_checkpoints/encoder_decoder.09-val_loss-0.1228-val_sad-176.7991-val_mse-0.0449.hdf5", custom_objects={"Unpooling": Unpooling()})

for layer in model.layers:
        layer.trainable = False


refinement = build_refinement(model)

PATIENCE=30
PATH_MODEL_CHECKPOINTS = "../model_checkpoints/"
PATH_LOGS = "../logs/"
BACKGROUNDS_PER_FG_TRAIN = 100
BACKGROUNDS_PER_FG_TEST = 20
BATCH_SIZE=16

base = "/home/devthebear/ds_uncombined/"
base_train = base + "train_ds/"
base_test  = base + "test_ds/"

fg_train = base_train + "all_foregrounds/"
bg_train = base_train + "train_selected/"
a_train = base_train + "all_alphas/"

fg_names_train = base_train + "training_fg_names.txt"
bg_names_train = base_train + "training_bg_names.txt"

fg_test = base_test + "fg/"
bg_test = base_test + "test_selected/"
a_test = base_test + "alpha/"

fg_names_test = base_test + "test_fg_names.txt"
bg_names_test = base_test + "test_bg_names.txt"

train_gen = DataGenerator(fg_names_train, bg_names_train, BACKGROUNDS_PER_FG_TRAIN, fg_train, bg_train, a_train, BATCH_SIZE, True)
test_gen = DataGenerator(fg_names_test, bg_names_test, BACKGROUNDS_PER_FG_TEST, fg_test, bg_test, a_test, BATCH_SIZE, True)


tensor_board = keras.callbacks.TensorBoard(log_dir=PATH_LOGS, histogram_freq=0, write_graph=True, write_images=True)
model_name = PATH_MODEL_CHECKPOINTS + 'refinement.{epoch:02d}-val_loss-{val_loss:.4f}-val_sad-{val_sad:.4f}-val_mse-{val_mse:.4f}.hdf5'
# model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True)
model_checkpoint = AltModelCheckpoint(filepath=model_name, alternate_model=refinement,  monitor='val_loss', verbose=1, save_best_only=True)

early_stop = EarlyStopping('val_loss', patience=6, verbose=1)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=3, verbose=1)

callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

multi_gpu_model = multi_gpu_model(refinement, gpus=2)


# In[214]:


inputs = refinement.input
optimizer = Adam(lr=0.00001)
multi_gpu_model.compile(optimizer=optimizer, loss=alpha_loss_wrapper(inputs), metrics=[sad_wrapper(inputs), mse_wrapper(inputs)])


# In[215]:


history = refinement.fit_generator(train_gen, validation_data=test_gen, use_multiprocessing=True, workers=3, callbacks=callbacks, shuffle=True, epochs=200)

