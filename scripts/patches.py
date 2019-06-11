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

