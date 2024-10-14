# NOT DONE IN THE PAPER:
# Reduce the complete Dataset with the Hash Method (not in each Volume)
# Can be used for all CT Datasets with alle slices in PNG Format in one Folder
# Needs longer due to doing it not only in the volumes (pairwise Comparison of all images in the Data)

# Part of the Code is from: https://github.com/volume-em/cem-dataset
# License: BSD 3-Clause License; Copyright (c) 2020, volume-em. All rights reserved.

import imagehash # hash
import glob # read in pathes
import shutil # copy files
from PIL import Image
import numpy as np
from argparse import ArgumentParser

# - shash = Difference hashing
#   1. Input image → greyscale image + reduced to 9×8
#   2. the first 8 pixels of each row are viewed in rows from left to right and compared with their right-hand neighbour
#   3. larger than right neighbour → 1 | smaller than right neighbour → 0
#       → 64 bit hash: 1111000000110000101110001100111010000110010011001000111010001110

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

    args = parser.parse_args()

    images_folder = args.images_folder
    save_folder = args.save_folder
    boarder = args.boarder


    # Create hashes -----------------------------------------------------------------------------------
    Ordner = sorted(glob.glob(images_folder + "/*"))  # Liste: Alle Pfade aus dem Ordner
    hash_array = []
    image_names = []
    for image_path in Ordner:  # durchläuft alle Pfade im Ordner

        name = image_path.split("/")[-1]  # Name des Bildes
        #print("Name: " + name)
        image_names.append(image_path) # Speichert den kompletten Pfad des Bildes in Liste

        hash = calculate_hash(image_path) # Methode erzegt den Hash (länge 64)
        #print(hash)
        hash_array.append(hash) # Hash in Liste speichern

    # Numpy array with Hashes: size [Number of Images x 64]
    hash_array = np.asarray(hash_array)
    print("final hash array:")
    print(hash_array.shape)
    print(hash_array)


    # Select different images  -------------------------------------------------------------------------
    print("--------------------------------")
    print("--------------------------------")

    exemplars = []

    while len(hash_array) > 0:

        # the reference hash is the first one in the list of remaining hashes
        ref_hash = hash_array[0]
        # ex. [1 0 1 0 ...]


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
        #print("sum", sum)


        # -- Hamming Distance < 12 ⇒ similar images --

        # if less than 12 -> then similar (at least 12 must be different)
        is_similar = sum <= boarder
        # z.b. [True False True True False]

        # Find a place from the similar
        matches = np.where(is_similar)[0]
        # ex. [0, 2, 3]
        #print("matches", matches)


        # -- Take the first match and save it in exemplars --

        # append image path of the first match
        exemplars.append(image_names[matches[0]])
        # = append image path of ref_hash


        # -- Save the match --

        # copies the image to a save_folder with the same name
        shutil.copy2(image_names[matches[0]], save_folder)  # target filename is /dst/dir/file.ext



        # -- Remove all matches in images pathes list and hashes list --

        # löscht alle die zu ähnlich zum ref_hash sind und auch den ref_hash selber
        hash_array = np.delete(hash_array, matches, axis=0)
        image_names = np.delete(image_names, matches, axis=0)

    # List of pathes with the not similar images that are left
    exemplars = np.asarray(exemplars)
    print("Left not similar images")
    print(exemplars.shape)
   # print(exemplars)



if __name__ == '__main__':
    main()