from keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rotation_range=5, 
                              zoom_range=0.1, 
                              horizontal_flip=True, 
                              width_shift_range=0.2, 
                              height_shift_range=0.2)