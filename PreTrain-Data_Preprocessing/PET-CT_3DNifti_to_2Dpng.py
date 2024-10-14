# 3D Nifti to 2D png File für AutoPET

import glob
import nibabel as nib
import cv2
import numpy as np



# ------------------------------------image normalization (for png/jepg)------------------------------------------
def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / (float(from_range)+0.00001), dtype=float)
    return to_min + (scaled * to_range)

# =============================================================================
# Save
# =============================================================================

def save(image, save_path1,patient_name,n):

    # --------------------------Image----------------------------------------------------
    # Nfti -> Numpy
    patient_pixels = nib.load(image)
    patient_pixels = patient_pixels.get_fdata()
    patient_pixels = patient_pixels.transpose(2, 1, 0)

    for i in range(patient_pixels.shape[0]):
        # 3D -> 2D
        img = patient_pixels[i, :, :]

        # Normalization for jpeg/png
        img = interval_mapping(img, img.min(), img.max(), 0, 255)

        # Save
        path = save_path1 + "/" + str(patient_name) + "_" + str(n) + "_" + str(i) + ".png"
        cv2.imwrite(path, img)

    print (i)



# ============================================================would =================
# Main
# =============================================================================
def main():

    # Daten: Images: DICOM Files in einem Ordner

    # ToDo: Path were the  DICOM files are saved:
    data_path = "/path/to/LIDC/manifest-1600709154662/LIDC-IDRI" # In this folder are many subfolders, one for each patient

    # ToDo: Path were the data should be saved
    save_path1 = "/folder/where/the/data/should/be/saved" # Create a folder on your computer where you want to save the pn images

    # ToDo: Choose a Window:
    #body_part = "abdomen": wl = 60, ww = 400
    #body_part == "angio": wl = 300, ww = 600
    #body_part == "bone": wl = 300, ww = 150
    #body_part == "brain": wl = 40, ww = 80
    #body_part == "chest": wl = 40, ww = 400
    #body_part == "lungs": wl = -400, ww = 1500
    wl = -400
    ww = 1500
    # We do not use this here, since we want to use all data for pre-training **

    # This loop saves one Nifti CT slice after another as png
    n=0
    Anzahl = 0
    Ordner = sorted(glob.glob(data_path + "/*"))  # List: All paths from the LIDC-IDRI folder (folder of the individual patients)
    for fileA in Ordner:  # Runs through all paths in the LIDC-IDRI folder (all patients)
        Patient_Name = fileA.split("/")[-1]  # Name of the patient

        print("Name: " + Patient_Name)

        Ordner2 = sorted(glob.glob(fileA + "/*")) # List: subfolder
        for fileB in Ordner2: # Runs through all pathes
            print("Unterordner: " + fileB.split("/")[-1])
            n = n+1

            Ordner3 = sorted(glob.glob(fileB + "/*"))  # List: different modalities (CT, PET,...)
            for fileC in Ordner3: # Runs through all different modalities (CT, PET,...)
                file_name = fileC.split("/")[-1]  # Name of the file (CT, PET,...)
                if file_name == "CT.nii.gz":
                    print(file_name)
                    save(fileC, save_path1, Patient_Name,n)
                    Anzahl += 1
                    print("Anzahl: ", Anzahl)



if __name__ == '__main__':
    main()