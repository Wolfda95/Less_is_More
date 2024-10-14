# Prprocessing

### 1. Download the Data
#### LIDC-IDRI Dataset
Download the LIDC-IDRI Dataset from here: [https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) 
#### PET-CT Dataset
Download the PET-CT Dataset from here: [https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/)

### 2. From DICOM or Nifti to PNG
The data comes as DICOM or Nifti images. We save each slice of each CT volume as a png file. We do not applay any windwoing since we want to use all data for pre-training. 

If you are using Conda on Linux, here is how to get started: 
1. Open your terminal and follow these steps: 
    1. <code>conda create --name Less_is_More_Preprocessing python==3.10</code>
    2. <code>conda activate Less_is_More_Preprocessing</code>
    4. <code>cd ...Less_is_More/PreTrain-Data_Preprocessing</code>
    5. <code>pip install -r requirements.txt</code>
2. Open LIDC_3DDICOM_to_2Dpng.py or PET-CT_3DNifti_to_2Dpng.py and adjust the folder pathes in the main method

