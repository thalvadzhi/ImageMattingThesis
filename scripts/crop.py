from CONSTANTS import TRIMAP_UNKNOWN_VALUE, IMG_HEIGHT, IMG_WIDTH
import numpy as np
import cv2 as cv

def select_crop_coordinates(trimap, crop_size=(IMG_WIDTH, IMG_HEIGHT)):
    '''
        Will return the coordinates of the topleft corner of a rectangle with size crop_size.
        The rectangle will be centered around a random value
    '''
    crop_w, crop_h = crop_size
    # get indices of all unknown pixels
    x_indices, y_indices = np.where(trimap == TRIMAP_UNKNOWN_VALUE)
    x, y = 0, 0
    n_unknown = len(y_indices)
    
    if n_unknown == 0:
        return x, y
    
    # pick one of the unknown pixels at random
    idx = np.random.choice(range(n_unknown))
    center_x = x_indices[idx]
    center_y = y_indices[idx]

    x = max(0, center_x - crop_w // 2)
    y = max(0, center_y - crop_h // 2)

    return x, y

def crop(img, x, y, crop_size=(IMG_WIDTH, IMG_HEIGHT)):
    crop_w, crop_h = crop_size

    padded_crop = np.zeros((crop_h, crop_w, 3), np.float32)
    crop = img[y:y + crop_h, x:x + crop_w]
    h, w = crop.shape[:2]
    padded_crop[0:h, 0:w] = crop
   return padded_crop
