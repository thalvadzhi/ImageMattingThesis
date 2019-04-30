from keras.utils import Sequence
import os 
from trimap import generate_trimap
from crop import crop_and_resize, select_crop_coordinates 
from CONSTANTS import IMG_HEIGHT, IMG_WIDTH
import re
from PIL import Image
import numpy as np
import random
import math

class DataGenerator(Sequence):
    IMG_EXTENSION_JPG = "jpg"
    IMG_EXTENSION_PNG = "png"


    def __init__(self, path_fg, path_bg, bgs_per_fg, base_path_fg, base_path_bg, base_path_alphas, batch_size, shuffle=True):
        self.path_fg = path_fg
        self.path_bg = path_bg
        self.base_path_fg = base_path_fg
        self.base_path_bg = base_path_bg
        self.base_path_alpha = base_path_alphas
       
        self.bgs_per_fg = bgs_per_fg # n backgrounds for one foreground
        self.fg, self.bg = self.__read_all_names()
        self.couples = self.__make_all_couples()
        self.n_images = len(self.couples)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()


    def __len__(self):
        return self.n_images // self.batch_size

    def __read_all_names(self):

        with open(self.path_fg) as f:
            fg = f.read().splitlines()
        
        with open(self.path_bg) as f:
            bg = f.read().splitlines()
        
        return fg, bg

    def __make_all_couples(self):
        couples = []
        global_counter = 0
        for fg in self.fg:
            for _ in range(self.bgs_per_fg):
                couples.append((fg, self.bg[global_counter]))
                global_counter += 1
        return couples
        

    def __crop_img_alpha_trimap(self, img, alpha, trimap, crop_size):
        x, y = select_crop_coordinates(trimap, crop_size)
        img_crop = crop_and_resize(img, x, y, crop_size)
        alpha_crop = crop_and_resize(alpha, x, y, crop_size)
        trimap_crop = crop_and_resize(trimap, x, y, crop_size)
        return img_crop, alpha_crop, trimap_crop

    def __crop_multiple_args(self, trimap, crop_size, *args):
        x, y = select_crop_coordinates(trimap, crop_size)
        trimap_crop = crop_and_resize(trimap, x, y, crop_size)
        crops = []
        for arg in args:
            crops.append(crop_and_resize(arg, x, y, crop_size))
        return (trimap_crop, *crops)

    def __randomly_flip_images(self,*args):
        
        if np.random.random_sample() > 0.5:
            flips = []
            for arg in args:
                flips.append(np.fliplr(arg))
            return flips
        return args

    

    def __process(self, fg_name, bg_name):
        im = Image.open(self.base_path_fg + fg_name)
        a = Image.open(self.base_path_alpha + fg_name)
        w, h = im.size
        bg = Image.open(self.base_path_bg + bg_name)
        bw, bh = bg.size
        wratio = w / bw
        hratio = h / bh

        if im.mode != 'RGB' and im.mode != 'RGBA':
            im = im.convert('RGB')
            
        if bg.mode != 'RGB':
            bg = bg.convert('RGB')
        
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = bg.resize((math.ceil(bw*ratio),math.ceil(bh*ratio)), Image.BICUBIC)

        # print("Alpha size={}, name={}, bg_name={}".format(a.size, fg_name, bg_name))
        return self.__composite5(im, bg, a, w, h)

    def __composite5(self, fg, bg, a, w, h):
        bg = bg.crop((0,0,w,h))
        
        fg_list = np.array(fg)
        bg_list = np.array(bg)
        alphas = np.array(a) / 255
        if len(alphas.shape) > 2:
            alphas = alphas[:, :, 0]
        
        if len(fg_list.shape) > 2:
            fg_list = fg_list[:, :, 0:3]

        alphas_w, alphas_h = alphas.shape
        alphas = alphas.reshape(alphas_w, alphas_h, 1)
        one_minus_alpha = 1 - alphas
        im = (alphas * fg_list + one_minus_alpha * bg_list).astype(np.uint8)
    
        return im, fg_list, bg_list, (alphas * 255).astype(np.uint8)

    def __getitem__(self, idx):
        #idx is the index of the batch

        index = idx * self.batch_size

        batch_names = self.couples[index:index+self.batch_size]
        batch_x = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 4), dtype=np.float32)
        batch_y = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 7), dtype=np.float32)

        crop_sizes = [(320, 320), (480, 480), (640, 640)]
        for name_index, (fg_name, bg_name) in enumerate(batch_names):
           
            composite, fg, bg, alpha = self.__process(fg_name, bg_name)
            crop_size = random.choice(crop_sizes)
            trimap = generate_trimap(alpha)

            trimap_crop, comp_crop, alpha_crop, fg_crop, bg_crop = self.__crop_multiple_args(trimap, crop_size, composite, alpha, fg, bg)
            
            trimap_crop, comp_crop, alpha_crop, fg_crop, bg_crop = self.__randomly_flip_images(trimap_crop, comp_crop, alpha_crop, fg_crop, bg_crop)


            
            batch_x[name_index, :, :, 0:3] = comp_crop / 255
            batch_x[name_index, :, :, 3] = trimap_crop / 255
            batch_y[name_index, :, :, 0] = alpha_crop[:, :, 0] / 255
            batch_y[name_index, :, :, 1:4] = fg_crop / 255
            batch_y[name_index, :, :, 4:7] = bg_crop / 255

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.couples)



