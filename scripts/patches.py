import numpy as np
import math

def pad_smaller_images(patch):
    w, h = patch.shape[:2]
    output = np.zeros((320, 320, 3))
    output[0:w, 0:h, :] = patch
    return output

def get_all_patches(image, trimap):
    '''
        Split the image in patches of 320x320px, predict an alpha matte for every one of them
    '''
    w, h = image.shape[:2]
    n_steps_w = math.ceil(w / 320)
    n_steps_h = math.ceil(h / 320)
    output = np.empty((n_steps_w * n_steps_h, 320, 320, 4))
    global_counter = 0
    for i in range(n_steps_w):
        for j in range(n_steps_h):
            patch_image = image[i * 320 : np.min([w, (i + 1) * 320]), j * 320: np.min([h, (j + 1) * 320]), :]
            patch_trimap = trimap[i * 320 : np.min([w, (i + 1) * 320]), j * 320: np.min([h, (j + 1) * 320]), :]
            
            if patch_image.shape != (320, 320, 3):
                print(patch_image.shape)
                patch_image = pad_smaller_images(patch_image)
                patch_trimap = pad_smaller_images(patch_trimap)
                print(patch_image.shape)

            
            patch_image = patch_image.reshape(1, 320, 320, 3)
            patch_trimap = patch_trimap[:, :, 0].reshape(1, 320, 320, 1)
            
            output[global_counter, :, :, 0:3] = patch_image / 255
            output[global_counter, :, :, 3:4] = patch_trimap / 255
            global_counter += 1
    return output, image.shape



def combine_patches(patches, original_shape):
    output = np.empty(original_shape)
    w, h = original_shape[:2]
    n_steps_w = math.ceil(w / 320)
    n_steps_h = math.ceil(h / 320)
    
    global_counter = 0
    for i in range(n_steps_w):
        for j in range(n_steps_h):
            patch = patches[global_counter][:, :, 0:len(original_shape)]
            global_counter += 1
            
            high_w = np.min([w, (i + 1) * 320]) - i * 320
            high_h =  np.min([h, (j + 1) * 320]) - j * 320
            output[i * 320 : np.min([w, (i + 1) * 320]), j * 320: np.min([h, (j + 1) * 320]), :] = patch[0:high_w, 0:high_h, 0:3]
    return output


def check_point_in_square(point, square):
    point_x, point_y = point
    x, y, width, height = square
    
    if point_x >= x and point_x <= x + width and point_y >= y and point_y <= y + height:
        return True
    else:
        return False

def check_point_in_squares(point, squares):
    for square in squares:
        if check_point_in_square(point, square) is True:
            return True
    return False

def make_new_square(center, img_width, img_height):
    center_x, center_y = center
    top_x = center_x - 160
    top_y = center_y - 160
    print(top_x, top_y, img_width, img_height)
    if top_x < 0:
        top_x = 0
    if top_y < 0:
        top_y = 0
    
    if top_x + 320 > img_width:
        how_much_greater = top_x + 320 - img_width 
        top_x -= how_much_greater
        
    if top_y + 320 > img_height:
        
        how_much_greater =  top_y + 320 - img_height
        print(how_much_greater)
        top_y -= how_much_greater
        
    return top_x, top_y, 320, 320

def patches_of_culture(img, trimap):
    img_width, img_height = img.shape[1], img.shape[0]
    y_indices, x_indices = np.where(trimap == 128)
    unknown_pixels  = list(zip(x_indices, y_indices))
    covered_pixels = set()
    squares = []
    for pixel in unknown_pixels:
        if check_point_in_squares(pixel, squares) is True:
            continue
        square = make_new_square(pixel, img_width, img_height)
        squares.append(square)
    return squares
    

def get_crops_by_squares(img, trimap, squares):
    crops = np.empty((len(squares), 320, 320, 4))
    for index, square in enumerate(squares):
        crops[index, :, :, 0:3] = img[square[1]:square[1]+320, square[0]:square[0]+320, :].reshape(1, 320, 320, 3) / 255
        crops[index, :, :, 3:4] = trimap[square[1]:square[1]+320, square[0]:square[0]+320, :].reshape(1, 320, 320, 1) / 255
    return crops

def combine_predictions(predictions, squares, original_shape):
    out = np.zeros((original_shape[0], original_shape[1], 1))
    mask = np.zeros((original_shape[0], original_shape[1], 1))
    for index, square in enumerate(squares):
        out[square[1]:square[1]+320, square[0]:square[0]+320, :] += predictions[index]
        mask[square[1]:square[1]+320, square[0]:square[0]+320, :] += (predictions[index] > 0)
    mask[mask == 0] = 1
    return out / mask
    
