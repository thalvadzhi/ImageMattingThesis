from keras.utils import Sequence
import os 
from trimap import generate_trimap
from crop import crop_and_resize, select_crop_coordinates 
from CONSTANTS import IMG_HEIGHT, IMG_WIDTH
import re
from PIL import Image
import numpy as np
import random

class DataGenerator(Sequence):
    IMG_EXTENSION_JPG = "jpg"
    IMG_EXTENSION_PNG = "png"

    def __init__(self, path_combined_imgs, path_alphas, batch_size, shuffle=True):
        self.combined_imgs_names = [img_name for img_name
                                    in os.listdir(path_combined_imgs)
                                    if img_name.endswith(self.IMG_EXTENSION_JPG) or img_name.endswith(self.IMG_EXTENSION_PNG)]
        
        self.n_images = len(self.combined_imgs_names)
        self.batch_size = batch_size
        self.path_combined_imgs = path_combined_imgs
        self.path_alphas = path_alphas
        self.shuffle = shuffle

        if self.shuffle==True:
            np.random.shuffle(self.combined_imgs_names)


    def __len__(self):
        return self.n_images // self.batch_size

    def __crop_img_alpha_trimap(self, img, alpha, trimap, crop_size):
        x, y = select_crop_coordinates(trimap, crop_size)
        img_crop = crop_and_resize(img, x, y, crop_size)
        alpha_crop = crop_and_resize(alpha, x, y, crop_size)
        trimap_crop = crop_and_resize(trimap, x, y, crop_size)
        return img_crop, alpha_crop, trimap_crop

    def __randomly_flip_images(self, img, alpha, trimap):
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            alpha = np.fliplr(alpha)
            trimap = np.fliplr(trimap)
        return img, alpha, trimap

    def __getitem__(self, idx):
        #idx is the index of the batch

        index = idx * self.batch_size

        batch_names = self.combined_imgs_names[index:index+self.batch_size]
        batch_x = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 4), dtype=np.float32)
        batch_y = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

        crop_sizes = [(320, 320), (480, 480), (640, 640)]
        for name_index, name in enumerate(batch_names):
            name_split = name.split("_")[:-1]
            extension = name.split(".")[-1]
            name_alpha = "_".join(name_split) + ".{}".format(extension)
            
            composite = np.array(Image.open(self.path_combined_imgs + name))
            alpha = np.array(Image.open(self.path_alphas + name_alpha))
            crop_size = random.choice(crop_sizes)
            # crop_size = (320, 320)
            trimap = generate_trimap(alpha)

            img_crop, alpha_crop, trimap_crop = self.__crop_img_alpha_trimap(composite, alpha, trimap, crop_size)
            
            img_crop, alpha_crop, trimap_crop = self.__randomly_flip_images(img_crop, alpha_crop, trimap_crop)
            
            batch_x[name_index, :, :, 0:3] = img_crop / 255
            batch_x[name_index, :, :, 3] = trimap_crop / 255
            batch_y[name_index, :, :, 0] = alpha_crop / 255

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.combined_imgs_names)



