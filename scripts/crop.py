from CONSTANTS import TRIMAP_UNKNOWN_VALUE, IMG_HEIGHT, IMG_WIDTH
import numpy as np
import cv2 as cv

def select_crop_coordinates(trimap, crop_size=(IMG_WIDTH, IMG_HEIGHT)):
    '''
        Will return the coordinates of the topleft corner of a rectangle with size crop_size.
        The rectangle will be centered around a random value
    '''
    crop_w, crop_h = crop_size
    # get indices of `all unknown pixels
    x_indices, y_indices = np.where(trimap / 255 == TRIMAP_UNKNOWN_VALUE)
    x, y = 0, 0
    n_unknown = len(y_indices)
    
    if n_unknown == 0:
        return x, y
    
    # pick one of the unknown pixels at random
    idx = np.random.choice(range(n_unknown))
    center_x = x_indices[idx]
    center_y = y_indices[idx]

    return center_y, center_x

def crop_by_coordinates(img, x, y, crop_size=(IMG_WIDTH, IMG_HEIGHT)):
    crop_w, crop_h = crop_size
    
    if len(img.shape) == 2:
        im_h, im_w = img.shape
        padded_crop = np.zeros((crop_h, crop_w), np.uint8)

    else:
        im_h, im_w, _ = img.shape
        padded_crop = np.zeros((crop_h, crop_w, 3), np.uint8)
    
    # first check if the x coordinate is in the first or the second half of the width
    # first calculate the size of the crop to the nearest side of the image
    # then calculate the size of the crop to the farthest side of the image
    # this way we guarantee that the crop is always the specified size and that the selected
    # unknown pixel is in the crop (might not be centered)
    if x < im_w // 2:
        x_low = max(0, x - crop_w//2)
        x_low_diff = x - x_low
        left_for_right = crop_w - x_low_diff
        x_high = min(im_w, x + left_for_right)
    else:
        x_high = min(im_w, x + crop_w // 2)
        x_high_diff = x_high - x
        left_for_left = crop_w - x_high_diff
        x_low = max(0, x - left_for_left)
    
    if y < im_h // 2:
        y_low = max(0, y - crop_h // 2)
        y_low_diff = y - y_low
        left_for_bottom = crop_h - y_low_diff
        y_high = min(im_h, y + left_for_bottom)
    else:
        y_high = min(im_h, y + crop_h // 2)
        y_high_diff = y_high - y
        left_for_top = crop_h - y_high_diff
        y_low = max(0, y - left_for_top)


    crop = img[y_low:y_high, x_low:x_high]
    h, w = crop.shape[:2]
    h_offset, w_offset = 0, 0
    
    padded_crop[h_offset:h+h_offset, w_offset:w+w_offset] = crop
    return padded_crop

def resize(img, size=(IMG_WIDTH, IMG_HEIGHT)):
    return cv.resize(img, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv.INTER_NEAREST)

def crop_and_resize(img, x, y, crop_size=(IMG_WIDTH, IMG_HEIGHT)):
    img_cropped = crop_by_coordinates(img, x, y, crop_size)
    if crop_size != (IMG_WIDTH, IMG_HEIGHT):
        img_cropped = resize(img_cropped, size=(IMG_WIDTH, IMG_HEIGHT))
    return img_cropped
