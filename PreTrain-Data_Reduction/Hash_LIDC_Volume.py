# PAPER:
# Reduces each Volume of the LIDC Dataset with the Hash approch
# Can only be used with the LIDC Dataset were the slices are stored in PNG Fomrat the following way:
# LIDC-IDRI-0001_0.png; LIDC-IDRI-0001_1.png; ... ; LIDC-IDRI-0002_133.png; LIDC-IDRI-0002_134.png; ... ;LIDC-IDRI-0003_394.png; LIDC-IDRI-0003_395.png

# Part of the Code is from: https://github.com/volume-em/cem-dataset
# License: BSD 3-Clause License; Copyright (c) 2020, volume-em. All rights reserved.

import imagehash # hash
import glob # read in pathes
import shutil # copy files
from PIL import Image
import numpy as np
from argparse import ArgumentParser


def calculate_hash(image, hash_size=8):
    pil_image = Image.open(image)
    hash = imagehash.dhash(pil_image, hash_size=hash_size).hash # size 8x8
    hash = np.ravel(hash) # flatten -> size 64x1
    return hash

def main():

    # Parser
    parser = ArgumentParser()
    parser.add_argument("--images_folder", default="/home/wolfda/Data/PreTrain_Lung/Test_LIDC/Data", type=str, help="Path to images")
    parser.add_argument("--save_folder", default="/home/wolfda/Data/PreTrain_Lung/Test_LIDC/NotSim", type=str, help="Path were images should be stored")
    parser.add_argument("--boarder", default=6, type=float ,help="")
    parser.add_argument("--Number_of_volumes", default=1010, type=float, help="1010")
    parser.add_argument("--Number_of_Images", default=243958, type=float, help="243958")

    args = parser.parse_args()

    images_folder = args.images_folder
    save_folder = args.save_folder
    boarder = args.boarder
    vol = args.Number_of_volumes
    num = args.Number_of_Images


    # Image Pathes in one List  ----------------------------------
    image_names = sorted(glob.glob(images_folder + "/*"))  # List: All paths from the folder

    x = 0
    image_volume_names = []
    hash_array = []
    exemplars = []
    # -------------- Passes through volume -------------------------
    for i in range(1, vol+1):  #  Number of volumes in the dataset
        print(i)

        # -------------------- Creates list with images of a volume --------------------
        while image_names[x].split("I-")[-1].split("_")[0] == str(i).zfill(4): #################### This is LIDC specific (so you can see where a volume ends)
            image_volume_names.append(image_names[x])
            hash = calculate_hash(image_names[x])  # method generates the hash (length 64)
            hash_array.append(hash)  # Save hash in list
            x += 1
            if x == num:  # Total number of images in the dataset
                break
        hash_array = np.asarray(hash_array)

        #--------------------- Passes through a volume  --------------------------------------
        # image_volume_names: Pathes of one volume

        while len(image_volume_names) > 0:

            # the reference hash is the first one in the list of remaining hashes
            ref_hash = hash_array[0]

            # -- Calculate Hamming Distance between the first hash and all other hashes --
            # Hamming Distance: Number of different digits || For binary: XOR operation (00→0, 11→0, 01→1, 10→1) + summing the resultant ones
            # → here in range [0.64] per image pair

            # XOR between ref_hash and all other hashes
            xor = np.logical_xor(ref_hash, hash_array)
            # [1 0 1 0 ...] -- [1 0 1 0 ...]
            #               -- [1 1 0 0 ...]
            #               -- [1 0 0 0 ...]
            #               -- [1 1 0 1 ...]
            #               -- [1 0 1 1 ...]

            # Sum of each XOR that was computed
            sum = xor.sum(1)
            # ex. [0, 50, 10, 8, 50]
            # -> how many bits are different per comparable image
            # -> the larger, the more different
            # print("sum", sum)

            # -- Hamming Distance < 12 ⇒ similar images --

            # if less than 12 -> then similar (at least 12 must be different)
            is_similar = sum <= boarder
            # ex. [True False True True False]

            # Find a place from the similar
            matches = np.where(is_similar)[0]
            # ex. [0, 2, 3]
            # print("matches", matches)

            # append image path of the first match
            exemplars.append(image_volume_names[matches[0]])
            # = append image path of ref_hash

            # copies the image to a save_folder with the same name
            shutil.copy2(image_volume_names[matches[0]], save_folder)  # target filename is /dst/dir/file.ext

            # deletes all that are too similar to the ref_hash and also the ref_hash itself
            hash_array = np.delete(hash_array, matches, axis=0)
            image_volume_names = np.delete(image_volume_names, matches, axis=0)

        # ------------ Delete lists of the volume ----------------------
        exemplars = np.asarray(exemplars)
        print("Left not similar images")
        print(exemplars.shape)
        #print(exemplars)
        image_volume_names = []
        exemplars = []
        hash_array = []




if __name__ == '__main__':
    main()