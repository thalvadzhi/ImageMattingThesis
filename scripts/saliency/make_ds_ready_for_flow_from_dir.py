# from keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import os

def load_train_test_names(path_train_names, path_test_names):
    with open(path_test_names, "r") as f:
        test_names = f.read().splitlines()
    
    with open(path_train_names, "r") as f:
        train_names = f.read().splitlines()

    return train_names, test_names

train_names, test_names = load_train_test_names( "C:\\Workspace\\ThesisImageMatting\\data\\saliency\\train_names.txt", "C:\\Workspace\\ThesisImageMatting\\data\\saliency\\test_names.txt")

def make_ds_flowable(train_names, test_names, path_images):
   
    for index, name in enumerate(train_names):
        print("\rCopying train {}\{}".format(index, len(train_names)), end="")
        copyfile(path_images + "images\\" + name, path_images + "train_ds\\images\\" + name)
        name_gt = os.path.splitext(name)[0] + ".png"
        copyfile(path_images + "ground_truth\\" + name_gt, path_images + "train_ds\\ground_truth\\" + name_gt)
        train_y.append(cv.imread(path_gt + filename + ".png"))
    for index, name in enumerate(test_names):
        print("\rCopying test {}\{}".format(index, len(test_names)), end="")
        copyfile(path_images + "images\\" + name, path_images + "test_ds\\images\\" + name)
        name_gt = os.path.splitext(name)[0] + ".png"
        copyfile(path_images + "ground_truth\\" + name_gt, path_images + "test_ds\\ground_truth\\" + name_gt)


make_ds_flowable(train_names, test_names, "D:\\saliency dataset\\complete\\")