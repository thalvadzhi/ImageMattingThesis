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

def build_saliency_model():
    vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(448, 320, 3))
    filters_init = 256
    #maybe reduce here?
    #block6
    pool5 = vgg.get_layer("block5_pool")

    conv1_side_pool5 = Conv2D(filters=filters_init, kernel_size=(7, 7), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv1_side_pool5")(pool5.output)
    conv2_side_pool5 = Conv2D(filters=filters_init, kernel_size=(7, 7), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv2_side_pool5")(conv1_side_pool5)
    conv3_side_pool5 = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation=None, name="conv3_side_pool5")(conv2_side_pool5)
    score_side_pool5_up = Conv2DTranspose(filters=1, kernel_size=64, strides=32, name="upsampling_side_pool5", activation="sigmoid", padding="same")(conv3_side_pool5)
    # score_side_pool5_up = deconv2d(conv3_side_pool5, 64, 32 ,"upsampling_side_pool5", output_shape)

    #block5
    block5_conv3 = vgg.get_layer("block5_conv3")

    conv1_side5 = Conv2D(filters=filters_init, kernel_size=(5, 5), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv1_side5")(block5_conv3.output)
    conv2_side5 = Conv2D(filters=filters_init, kernel_size=(5, 5), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv2_side5")(conv1_side5)
    conv3_side5 = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation=None, name="conv3_side5")(conv2_side5)
    score_side5_up = Conv2DTranspose(filters=1, kernel_size=32, strides=16, name="upsampling_side5", activation="sigmoid", padding="same")(conv3_side5)
    # score_side5_up = deconv2d(conv3_side5, 32, 16, "upsampling_side5", output_shape)

    #block4
    block4_conv3 = vgg.get_layer("block4_conv3")

    conv1_side4 = Conv2D(filters=filters_init//2, kernel_size=(5, 5), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv1_side4")(block4_conv3.output)
    conv2_side4 = Conv2D(filters=filters_init//2, kernel_size=(5, 5), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv2_side4")(conv1_side4)
    conv3_side4 = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation=None, name="conv3_side4")(conv2_side4)
    score_side_pool5_up_4 = Conv2DTranspose(filters=1, kernel_size=8, strides=4, name="upsampling_c3_sp5_4", padding="same")(conv3_side_pool5)
    # score_side_pool5_up_4 = deconv2d(conv3_side_pool5, 8, 4, "upsampling_c3_sp5_4", K.shape(conv3_side4))

    score_side5_up_4 = Conv2DTranspose(filters=1, kernel_size=4, strides=2, name="upsampling_c3_s5_4", padding="same")(conv3_side5)
    # score_side5_up_4 = deconv2d(conv3_side5, 4, 2, "upsampling_c3_s5_4", K.shape(conv3_side4))

    concat_side4 = Concatenate(name="concat_s4")([conv3_side4, score_side_pool5_up_4, score_side5_up_4])
    conv4_side4 = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation=None, name="conv4_side4")(concat_side4)
    score_side4_up = Conv2DTranspose(filters=1, kernel_size=16, strides=8, name="upsampling_concat4", activation="sigmoid", padding="same")(conv4_side4)
    # score_side4_up = deconv2d(conv4_side4, 16, 8, "upsampling_concat4", output_shape)

    #block3
    block3_conv3 = vgg.get_layer("block3_conv3")

    conv1_side3 = Conv2D(filters=filters_init//2, kernel_size=(5, 5), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv1_side3")(block3_conv3.output)
    conv2_side3 = Conv2D(filters=filters_init//2, kernel_size=(5, 5), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv2_side3")(conv1_side3)
    conv3_side3 = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation=None, name="conv3_side3")(conv2_side3)

    score_side_pool5_up_3 = Conv2DTranspose(filters=1, kernel_size=16, strides=8, name="upsampling_c3_sp5_3", padding="same")(conv3_side_pool5)
    # score_side_pool5_up_3 = deconv2d(conv3_side_pool5, 16, 8, "upsampling_c3_sp5_3", K.shape(conv3_side3))

    score_side5_up_3 = Conv2DTranspose(filters=1, kernel_size=8, strides=4, name="upsampling_c3_s5_3", padding="same")(conv3_side5)
    # score_side5_up_3 = deconv2d(conv3_side5, 8, 4, "upsampling_c3_s5_3", K.shape(conv3_side3))

    concat_side3 = Concatenate(name="concat_side3")([conv3_side3, score_side_pool5_up_3, score_side5_up_3])
    conv4_side3 = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation=None, name="conv4_side3")(concat_side3)
    score_side3_up = Conv2DTranspose(filters=1, kernel_size=8, strides=4, name="upsampling_concat3", activation="sigmoid", padding="same")(conv4_side3)
    # score_side3_up = deconv2d(conv4_side3, 8, 4, "upsampling_concat3", output_shape)

    #block2
    block2_conv2 = vgg.get_layer("block2_conv2")
    conv1_side2 = Conv2D(filters=filters_init//4, kernel_size=(3, 3), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv1_side2")(block2_conv2.output)
    conv2_side2 = Conv2D(filters=filters_init//4, kernel_size=(3, 3), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv2_side2")(conv1_side2)
    conv3_side2 = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation=None, name="conv3_side2")(conv2_side2)

    score_side_pool5_up_2 = Conv2DTranspose(filters=1, kernel_size=32, strides=16, name="upsampling_c3_sp5_2", padding="same")(conv3_side_pool5)
    # score_side_pool5_up_2 = deconv2d(conv3_side_pool5, 32, 16, "upsampling_c3_sp5_2", K.shape(conv3_side2))

    score_side5_up_2 = Conv2DTranspose(filters=1, kernel_size=16, strides=8, name="upsampling_c3_s5_2", padding="same")(conv3_side5)
    # score_side5_up_2 = deconv2d(conv3_side5, 16, 8, "upsampling_c3_s5_2", K.shape(conv3_side2))

    score_side4_up_2 = Conv2DTranspose(filters=1, kernel_size=8, strides=4, name="upsampling_c3_s4_2", padding="same")(conv3_side4)
    #  score_side4_up_2 = deconv2d(conv3_side4, 8, 4, "upsampling_c3_s4_2", K.shape(conv3_side2))

    score_side3_up_2 = Conv2DTranspose(filters=1, kernel_size=4, strides=2, name="upsampling_c3_s3_2", padding="same")(conv3_side3)
    # score_side3_up_2 = deconv2d(conv3_side3, 4, 2, "upsampling_c3_s3_2", K.shape(conv3_side2))

    concat_side2 = Concatenate(name="concat_side2")([conv3_side2, score_side_pool5_up_2, score_side5_up_2, score_side4_up_2, score_side3_up_2])

    conv4_side2 = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation=None, name="conv4_side2")(concat_side2)
    score_side2_up = Conv2DTranspose(filters=1, kernel_size=4, strides=2, name="upsampling_concat2", activation="sigmoid", padding="same")(conv4_side2)
    # score_side2_up = deconv2d(conv4_side2, 4, 2, "upsampling_concat2", output_shape)

    #block1
    block1_conv2 = vgg.get_layer("block1_conv2")
    conv1_side1 = Conv2D(filters=filters_init//4, kernel_size=(3, 3), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv1_side1")(block1_conv2.output)
    conv2_side1 = Conv2D(filters=filters_init//4, kernel_size=(3, 3), strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv2_side1")(conv1_side1)
    conv3_side1 = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation=None, name="conv3_side1")(conv2_side1)

    score_side_pool5_up_1 = Conv2DTranspose(filters=1, kernel_size=64, strides=32, name="upsampling_c3_sp5_1", padding="same")(conv3_side_pool5)
    # score_side_pool5_up_1 = deconv2d(conv3_side_pool5, 64, 32, "upsampling_c3_sp5_1", K.shape(conv3_side1))

    score_side5_up_1 = Conv2DTranspose(filters=1, kernel_size=32, strides=16, name="upsampling_c3_s5_1", padding="same")(conv3_side5)
    # score_side5_up_1 = deconv2d(conv3_side5, 32, 16, "upsampling_c3_s5_1", K.shape(conv3_side1))

    score_side4_up_1 = Conv2DTranspose(filters=1, kernel_size=16, strides=8, name="upsampling_c3_s4_1", padding="same")(conv3_side4)
    # score_side4_up_1 = deconv2d(conv3_side4, 16, 8, "upsampling_c3_s4_1", K.shape(conv3_side1))

    score_side3_up_1 = Conv2DTranspose(filters=1, kernel_size=8, strides=4, name="upsampling_c3_s3_1", padding="same")(conv3_side3)
    # score_side3_up_1 = deconv2d(conv3_side3, 8, 4, "upsampling_c3_s3_1", K.shape(conv3_side1))

    concat_side1 = Concatenate(name="concat_side1")([conv3_side1, score_side_pool5_up_1, score_side5_up_1, score_side4_up_1, score_side3_up_1])
    score_side1_up = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation="sigmoid", name="conv4_side1")(concat_side1)

    #fusing

    concat_fusion = Concatenate(name="concat_fusion")([score_side1_up, score_side2_up, score_side3_up, score_side4_up, score_side5_up, score_side_pool5_up])
    # concat_fusion = tf.concat([score_side1_up, score_side2_up, score_side3_up, score_side4_up, score_side5_up, score_side_pool5_up], axis=-1, name="concat_fusion")
    score_fusion = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer="he_normal", activation="sigmoid", name="conv_fuse")(concat_fusion)

    model = Model(inputs=vgg.input, outputs=[score_side1_up, score_side2_up, score_side3_up, score_side4_up, score_side5_up, score_side_pool5_up, score_fusion])

    return model, [score_side1_up, score_side2_up, score_side3_up, score_side4_up, score_side5_up, score_side_pool5_up, score_fusion]

