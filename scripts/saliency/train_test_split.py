from sklearn.model_selection import train_test_split
import os
import numpy as np

path_base = "..\\..\\data\\"
path_imgs = path_base + "complete\\images\\"
path_gt = path_base + "complete\\ground_truth\\"

imgs = os.listdir(path_imgs)
def get_classes(imgs):
    classes = []
    for img in imgs:
        if "msra" in img:
            classes.append(0)
        elif "hku" in img:
            classes.append(1)
        elif "ecssd" in img:
            classes.append(2)
    return classes
classes = get_classes(imgs)
# print(len(classes))
imgs_train, imgs_test = train_test_split(imgs, stratify=classes, train_size=0.9, random_state=42)

output_train_file_name = "train_names.txt"
output_test_file_name = "test_names.txt"
print(len(imgs_test))
with open(output_test_file_name, "w") as f:
    for name in imgs_test:
        f.write("{}\n".format(name))

with open(output_train_file_name, "w") as f:
    for name in imgs_train:
        f.write("{}\n".format(name))
