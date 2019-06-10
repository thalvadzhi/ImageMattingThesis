from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import os

def generate_generator(gen):
    while True:
        X, y = next(gen)
        yield X, [y, y, y, y, y, y, y]

def get_data_generators(base="D:\\saliency dataset\\complete\\", batch_size=16, target_size=(448, 320)):
    seed = 42

    train_img = ImageDataGenerator(zoom_range=0.1, horizontal_flip=True, rescale=1.0/255.0)
    train_img_gen = train_img.flow_from_directory(base + "train_ds\\images", class_mode=None, seed=seed, batch_size=16, target_size=(448, 320))
    train_gt_gen = train_img.flow_from_directory(base + "train_ds\\ground_truth", color_mode="grayscale", class_mode=None, seed=seed, batch_size=16, target_size=(448, 320))
    train_gen = zip(train_img_gen, train_gt_gen)

    test_img = ImageDataGenerator(zoom_range=0.1, horizontal_flip=True, rescale=1.0/255.0)

    test_img_gen = test_img.flow_from_directory(base + "test_ds\\images", class_mode=None, seed=seed, batch_size=16, target_size=(448, 320))
    test_gt_gen = test_img.flow_from_directory(base + "test_ds\\ground_truth", color_mode="grayscale", class_mode=None, seed=seed, batch_size=16, target_size=(448, 320))
    test_gen = zip(test_img_gen, test_gt_gen)
    return generate_generator(train_gen), generate_generator(test_gen)
