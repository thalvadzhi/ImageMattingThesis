import os
from shutil import copyfile

base = "..\\..\\data\\saliency dataset\\"
msra = base + "MSRA10K_Imgs_GT\\Imgs\\"
hku = base + "HKU-IS\\"
ecssd = base + "ecssd\\"

hku_gt = hku + "gt\\"
hku_img = hku + "imgs\\"

ecssd_gt = ecssd + "ground_truth_mask\\"
ecssd_img = ecssd + "images\\"

destination_imgs = base + "complete\\images\\"
destination_gt = base + "complete\\ground_truth\\"

files_msra = os.listdir(msra)
for index, f in enumerate(files_msra):
    print("\rCopying {}\{}".format(index, len(files_msra)), end="")
    file_no_extension = os.path.splitext(f)[0]
    if f.endswith(".jpg"):
        copyfile(msra + f, destination_imgs + file_no_extension + "_msra.jpg")
    elif f.endswith(".png"):
        copyfile(msra + f, destination_gt + file_no_extension + "_msra.png")

files_hku_img = os.listdir(hku_img)
files_hku_gt = os.listdir(hku_gt)

for index, f in enumerate(files_hku_img):
    print("\rCopying {}\{}".format(index + 1, len(files_hku_img)), end="")

    file_no_extension = os.path.splitext(f)[0]
    extension = os.path.splitext(f)[1]
    copyfile(hku_img + f, destination_imgs + file_no_extension + "_hku" + extension)
    

for index, f in enumerate(files_hku_gt):
    print("\rCopying {}\{}".format(index + 1, len(files_hku_gt)), end="")

    file_no_extension = os.path.splitext(f)[0]
    extension = os.path.splitext(f)[1]
    copyfile(hku_gt + f, destination_gt + file_no_extension + "_hku" + extension)
    
files_ecssd_img = os.listdir(ecssd_img)
files_ecssd_gt = os.listdir(ecssd_gt)

for index, f in enumerate(files_ecssd_img):
    print("\rCopying {}\{}".format(index + 1, len(files_ecssd_img)), end="")

    file_no_extension = os.path.splitext(f)[0]
    extension = os.path.splitext(f)[1]
    copyfile(ecssd_img + f, destination_imgs + file_no_extension + "_ecssd" + extension)
    

for index, f in enumerate(files_ecssd_gt):
    print("\rCopying {}\{}".format(index + 1, len(files_ecssd_gt)), end="")

    file_no_extension = os.path.splitext(f)[0]
    extension = os.path.splitext(f)[1]
    copyfile(ecssd_gt + f, destination_gt + file_no_extension + "_ecssd" + extension)

dest_img = os.listdir(destination_imgs)
dest_gt = os.listdir(destination_gt)

for index, f in enumerate(dest_img):
    print("\rRenaming {}\{}".format(index + 1, len(dest_img)), end="")
    if ".." in f:
        os.rename(destination_imgs + f, destination_imgs + f.replace("..", "."))

for index, f in enumerate(dest_gt):
    print("\rRenaming {}\{}".format(index + 1, len(dest_gt)), end="")
    if ".." in f:
        os.rename(destination_gt + f, destination_gt + f.replace("..", "."))