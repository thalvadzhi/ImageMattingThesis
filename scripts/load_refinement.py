import keras
from keras.models import Model, Sequential
from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D, BatchNormalization, Reshape, Concatenate, ZeroPadding2D
import numpy as np
import keras.backend as K
from unpooling import Unpooling


def build_refinement(encoder_decoder_model):
    #remove the trimap from the input
    input_no_trimap = Lambda(lambda i: i[:, :, :, 0:3])(encoder_decoder_model.input)

    #append the output alpha matte of the encoder_decoder stage
    x = Concatenate(axis=3, name="concatenate_refinement")([input_no_trimap, encoder_decoder_model.output])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros',name="conv_refinement1")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros',name="conv_refinement2")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros',name="conv_refinement3")(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='refinement_output', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)

    model = Model(inputs=encoder_decoder_model.input, outputs=x)
    return model
