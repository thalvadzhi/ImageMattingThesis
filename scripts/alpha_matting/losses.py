import keras.backend as K
from scripts.CONSTANTS import EPS, EPS_SQ, TRIMAP_UNKNOWN_VALUE


def alpha_loss_wrapper(input_tensor):
    def alpha_loss(y_true, y_pred):
        trimap = input_tensor[:, :, :, 3]
        # trimap has 3 values : 0 for bg, 255 for fg, and 128 for uncertain areas. Mask will be 1 for uncertain areas and 0 for all others
        mask = K.cast(K.equal(trimap, TRIMAP_UNKNOWN_VALUE), "float32")
        shape_y_pred = K.int_shape(y_pred)
        mask = K.reshape(mask, (-1, int(shape_y_pred[1]), int(shape_y_pred[2]), 1))
        diff = y_pred - y_true[:, :, :, :1]
        # we're only interested in the uncertain areas. It is clear what to do in all others
        diff *= mask
        n_pixels = K.sum(mask)
        return K.sum(K.sqrt(K.square(diff) + EPS_SQ)) / (n_pixels + EPS)
    return alpha_loss


def compositional_loss_wrapper(input_tensor):
    def compositional_loss(y_true, y_pred):
        trimap = input_tensor[:, :, :, 3]
        original_image = input_tensor[:, :, :, 0:3]
        mask = K.cast(K.equal(trimap, TRIMAP_UNKNOWN_VALUE), "float32")
        shape_y_pred = K.int_shape(y_pred)
        mask = K.reshape(mask, (-1, int(shape_y_pred[1]), int(shape_y_pred[2]), 1))
        fg = y_true[:, :, :, 1:4]
        bg = y_true[:, :, :, 4:7]
        predicted_image = y_pred * fg + (1 - y_pred) * bg
        diff = predicted_image - original_image
        diff *= mask
        n_pixels = K.sum(mask)
        return K.sum(K.sqrt(K.square(diff) + EPS_SQ)) / (n_pixels + EPS)
    return compositional_loss


def overall_loss_wrapper(input_tensor):
    def overall_loss(y_true, y_pred):
        alpha_loss = alpha_loss_wrapper(input_tensor)
        compositional_loss = compositional_loss_wrapper(input_tensor)
        w_l = 0.5
        return w_l * alpha_loss(y_true, y_pred) + \
            (1 - w_l) * compositional_loss(y_true, y_pred)
    return overall_loss


def mse_wrapper(input_tensor):
    def mse(y_true, y_pred):
        trimap = input_tensor[:, :, :, 3]
        diff = y_true[:, :, :, :1] - y_pred
        mask = K.cast(K.equal(trimap, TRIMAP_UNKNOWN_VALUE), "float32")
        shape_y_pred = K.int_shape(y_pred)
        mask = K.reshape(mask, (-1, int(shape_y_pred[1]), int(shape_y_pred[2]), 1))
        diff *= mask
        n_pixels = K.sum(mask)
        return K.sum(K.square(diff) + EPS_SQ) / (n_pixels + EPS)
    return mse

def sad_wrapper(input_tensor):
    def sad(y_true, y_pred):
        trimap = input_tensor[:, :, :, 3]
        diff = K.abs(y_true[:, :, :, :1] - y_pred)
        mask = K.cast(K.equal(trimap, TRIMAP_UNKNOWN_VALUE), "float32")
        shape_y_pred = K.int_shape(y_pred)
        mask = K.reshape(mask, (-1, int(shape_y_pred[1]), int(shape_y_pred[2]), 1))
        diff *= mask
        loss = K.sum(diff)
        # loss is scaled by 1000 due to large n of images
        return loss / 1000
    return sad