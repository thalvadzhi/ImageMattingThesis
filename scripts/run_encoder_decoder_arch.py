#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('load_ext', 'autoreload')#
#get_ipython().run_line_magic('autoreload', '2')
import sys
import keras
from alt_model_checkpoint import AltModelCheckpoint
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D
sys.path.insert(0, "../scripts/")
from load_encoder_decoder import build_encoder_decoder_from_vgg
from data_generator import DataGenerator
from losses import overall_loss_wrapper
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.optimizers import Adam

# In[8]:


PATIENCE=30
PATH_MODEL_CHECKPOINTS = "../model_checkpoints/"
PATH_LOGS = "../logs"
HOME = "/home/devthebear/"
BATCH_SIZE=32


# In[4]:


inputs, model = build_encoder_decoder_from_vgg()


# In[9]:


path_combined_train = HOME + "ds/train_ds/merged_ds_two/"
path_alphas_train = HOME + "ds/train_ds/all_alphas/"

path_combined_test = HOME + "ds/test_ds/test_ds/" 
path_alphas_test = HOME + "ds/test_ds/alpha/" 


# In[7]:


#get_ipython().system('pwd')


# In[10]:



train_gen = DataGenerator(path_combined_train, path_alphas_train, BATCH_SIZE, True)
test_gen = DataGenerator(path_combined_test, path_alphas_test, BATCH_SIZE, True) 


# In[13]:


tensor_board = keras.callbacks.TensorBoard(log_dir=PATH_LOGS, histogram_freq=0, write_graph=True, write_images=True)
model_name = PATH_MODEL_CHECKPOINTS + 'encoder_decoder.{epoch:02d}-{val_loss:.4f}.hdf5'
model_name_multi_gpu = PATH_MODEL_CHECKPOINTS + 'encoder_decoder_multi_gpu.{epoch:02d}-{val_loss:.4f}.hdf5'
model_checkpoint = AltModelCheckpoint(filepath=model_name,alternate_model=model,  monitor='val_loss', verbose=1, save_best_only=True)
model_checkpoint_multigpu = ModelCheckpoint(model_name_multi_gpu, monitor='val_loss', verbose=1, save_best_only=True)
early_stop = EarlyStopping('val_loss', patience=PATIENCE, verbose=1)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=PATIENCE // 4, verbose=1)

callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]


# In[11]:


multi_gpu_model = multi_gpu_model(model, gpus=2)


# In[14]:

optimizer = Adam(lr=0.00001)
multi_gpu_model.compile(optimizer=optimizer, loss=overall_loss_wrapper(inputs))


# In[24]:


history = multi_gpu_model.fit_generator(train_gen, epochs=200, verbose=1, callbacks=callbacks, validation_data=test_gen, workers=3, use_multiprocessing=True)

