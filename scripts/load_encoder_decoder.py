import keras
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D, BatchNormalization, Reshape, Concatenate, ZeroPadding2D
import numpy as np
import keras.backend as K
from unpooling import Unpooling

def expand_dimension(old_weights):
    '''
        Function to expand the weights from (3, 3, 3, 64) to (3, 3, 4, 64) to accomodate the extra channel for the trimap
        The extra values are initialized with random numbers
    '''
    W, b = old_weights
    W_np = np.array(W)
    #initialize with random values (original paper uses zeros is this a problem???)
    W_new = np.random.rand(3, 3, 4, 64)
    #positions that are present in original values are preserved
    W_new[:, :, 0:3, :] = W_np
    return W_new, b

def copy_conv(layer, inp, weights):
    filters = layer.filters
    kernel_size = layer.kernel_size
    strides = layer.strides
    padding = layer.padding
    data_format = layer.data_format
    dilation_rate = layer.dilation_rate
    activation = layer.activation
    use_bias = layer.use_bias
    kernel_initializer = layer.kernel_initializer
    bias_initializer = layer.bias_initializer
    kernel_regularizer = layer.kernel_regularizer
    bias_regularizer = layer.bias_regularizer
    activity_regularizer = layer.activity_regularizer
    kernel_constraint = layer.kernel_constraint
    bias_constraint = layer.bias_constraint
    name = layer.name
    return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                  data_format=data_format, dilation_rate=dilation_rate, activation=activation, 
                  use_bias=use_bias, kernel_initializer=kernel_initializer, 
                  bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, 
                  bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, 
                  kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, name=name, weights=weights)(inp)

def copy_max_pool(layer, inp, weights):
    pool_size = layer.pool_size
    strides = layer.strides
    padding = layer.padding
    data_format = layer.data_format
    name = layer.name
    return MaxPooling2D(pool_size=pool_size, strides=strides, 
                        padding=padding, data_format=data_format, weights=weights, name=name)(inp)

def convert_dense_to_conv(layer, f_dim, inp):
    '''
        Convert the dense layer to a convolutional layer.
        layer is the dense layer
        f_dim is the input dimension of the flatten layer before the dense layer - (7, 7, 512) in the original
        inp is the last convolutional layer in the original network
    '''
    input_shape = layer.input_shape
    output_dim = layer.get_weights()[1].shape[0]
    W, b = layer.get_weights()
    shape = (f_dim[1], f_dim[2], f_dim[3], output_dim) # shape = (7, 7, 512, 4096)
    new_W = W.reshape(shape)
    return Conv2D(output_dim,
              (f_dim[1], f_dim[2]),
              strides=(1,1),
              activation=layer.activation,
              padding='valid',
              weights=[new_W, b])(inp)

def drop_last_layers_vgg(vgg):
    # drop the last two dense layers
    vgg.layers.pop()
    vgg.layers.pop()


def build_encoder(vgg):
    '''
        This function makes the following changes to the VGG16 network:
            1) Drops the final two dense layers
            2) Changes the input dimensions from (224, 224, 3) to (320, 320, 4)
            3) Changes the filter of the first conv layer to be of size (3, 3, 4), instead of (3, 3, 3) to accomodate for the extra channel (the trimap)
            4) Converts the first dense layer to a convolutional layer
    '''
    # make the input images be of size 320x320 and have 4 channels (last channel will be for the trimap)
    inputs = Input(shape=(320, 320, 4))
    #will contain all layers that are before pooling layers - this is needed for the Unpooling
    before_pooling = []
    x = inputs
    for layer in vgg.layers[1:]:
        weights = layer.get_weights()
        if "Conv2D" in str(layer):
            if "block1_conv1" == layer.name:
                # Changes the weights from shape (3, 3, 3) to (3, 3, 4) to accomodate trimap in the input layer
                weights = expand_dimension(weights)
            x = copy_conv(layer, x, weights)
        elif "MaxPooling2D" in str(layer):
            before_pooling.append(x)
            x = copy_max_pool(layer, x, weights)
        elif "Flatten" in str(layer):
            # record flatten's input shape
            f_dim = layer.input_shape
        elif "Dense" in str(layer):
            # convert the dense layer to a conv layer
            x = convert_dense_to_conv(layer, f_dim, x)
    return inputs, x, before_pooling

def build_decoder(x, before_pooling):
    # maybe try with transpose convolution instead of unpooling
    before_pooling_idx = len(before_pooling)
   
    # these are needed because of the dense layer converted to conv
    y = UpSampling2D(size=(2, 2))(x)
    y = ZeroPadding2D(padding=(1,1))(y)

    y = Conv2D(kernel_size=(1, 1), filters=512, padding="same", name="deconv6", activation="relu", kernel_initializer="he_normal")(y)
    y = BatchNormalization()(y)
    y = UpSampling2D(size=(2, 2), name="upsampling5")(y)

    before_pooling_idx -= 1
    original_shape = K.int_shape(before_pooling[before_pooling_idx])
    new_shape = (1, original_shape[1], original_shape[2], original_shape[3])
    before_pooling_reshaped = Reshape(new_shape)(before_pooling[before_pooling_idx])
    after_pooling_reshaped = Reshape(new_shape)(y)
    together = Concatenate(axis=1)([before_pooling_reshaped, after_pooling_reshaped])
    
    y = Unpooling()(together)

    y = Conv2D(kernel_size=(5, 5), filters=512, padding="same", name="deconv5", activation="relu", kernel_initializer="he_normal")(y)
    y = UpSampling2D(size=(2, 2), name="upsampling4")(y)

    before_pooling_idx -= 1
    original_shape = K.int_shape(before_pooling[before_pooling_idx])
    new_shape = (1, original_shape[1], original_shape[2], original_shape[3])
    before_pooling_reshaped = Reshape(new_shape)(before_pooling[before_pooling_idx])
    after_pooling_reshaped = Reshape(new_shape)(y)
    together = Concatenate(axis=1)([before_pooling_reshaped, after_pooling_reshaped])
    
    y = Unpooling()(together)

    y = Conv2D(kernel_size=(5, 5), filters=256, padding="same", name="deconv4", activation="relu", kernel_initializer="he_normal")(y) 
    y = UpSampling2D(size=(2, 2), name="upsampling3")(y)
    
    before_pooling_idx -= 1
    original_shape = K.int_shape(before_pooling[before_pooling_idx])
    new_shape = (1, original_shape[1], original_shape[2], original_shape[3])
    before_pooling_reshaped = Reshape(new_shape)(before_pooling[before_pooling_idx])
    after_pooling_reshaped = Reshape(new_shape)(y)
    together = Concatenate(axis=1)([before_pooling_reshaped, after_pooling_reshaped])
    
    y = Unpooling()(together)

    y = Conv2D(kernel_size=(5, 5), filters=128, padding="same", name="deconv3", activation="relu", kernel_initializer="he_normal")(y) 
    y = UpSampling2D(size=(2, 2), name="upsampling2")(y)

    before_pooling_idx -= 1
    original_shape = K.int_shape(before_pooling[before_pooling_idx])
    new_shape = (1, original_shape[1], original_shape[2], original_shape[3])
    before_pooling_reshaped = Reshape(new_shape)(before_pooling[before_pooling_idx])
    after_pooling_reshaped = Reshape(new_shape)(y)
    together = Concatenate(axis=1)([before_pooling_reshaped, after_pooling_reshaped])
    
    y = Unpooling()(together)

    y = Conv2D(kernel_size=(5, 5), filters=64, padding="same", name="deconv2", activation="relu", kernel_initializer="he_normal")(y)
    y = UpSampling2D(size=(2, 2), name="upsampling1")(y)

    before_pooling_idx -= 1
    original_shape = K.int_shape(before_pooling[before_pooling_idx])
    new_shape = (1, original_shape[1], original_shape[2], original_shape[3])
    before_pooling_reshaped = Reshape(new_shape)(before_pooling[before_pooling_idx])
    after_pooling_reshaped = Reshape(new_shape)(y)
    together = Concatenate(axis=1)([before_pooling_reshaped, after_pooling_reshaped])
    
    y = Unpooling()(together)

    y = Conv2D(kernel_size=(5, 5), filters=64, padding="same", name="deconv1", activation="relu", kernel_initializer="he_normal")(y)
    
    y = Conv2D(kernel_size=(5, 5), filters=1, padding="same", name="Raw_Alpha_Pred", activation="sigmoid", kernel_initializer="he_normal")(y)
    
    return y

def build_encoder_decoder(vgg):
    drop_last_layers_vgg(vgg)
    inputs, x, before_pooling = build_encoder(vgg)
    y = build_decoder(x, before_pooling)
    return inputs, Model(inputs=inputs, outputs=y)

def build_encoder_decoder_from_vgg():
    vgg = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
    return build_encoder_decoder(vgg)