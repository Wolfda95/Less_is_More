# Less is More: Selective reduction of CT data for self-supervised pre-training of deep learning models with contrastive learning improves downstream classification performance

Publication about self-supervised pre-training in medical imaging accepted in the Elsevier Journal [Computers in Biology and Medicine](https://www.sciencedirect.com/journal/computers-in-biology-and-medicine). 

DOI: [https://doi.org/10.1038/s41598-023-46433-0](https://doi.org/10.1016/j.compbiomed.2024.109242)

## Introduction
![image](https://github.com/user-attachments/assets/445bc797-62b9-472f-8173-518c0e042d86)

#### Background:
Self-supervised pre-training of deep learning models with contrastive learning is a widely used technique in image analysis. Current findings indicate a strong potential for contrastive pre-training on medical images. However, further research is necessary to incorporate the particular characteristics of these images.
#### Method:
We hypothesize that the similarity of medical images hinders the success of contrastive learning in the medical imaging domain. To this end, we investigate different strategies based on deep embedding, information theory, and hashing in order to identify and reduce redundancy in medical pre-training datasets. The effect of these different reduction strategies on contrastive learning is evaluated on two pre-training datasets and several downstream classification tasks.
#### Results:
In all of our experiments, dataset reduction leads to a considerable performance gain in downstream tasks, e.g., an AUC score improvement from 0.78 to 0.83 for the COVID CT Classification Grand Challenge, 0.97 to 0.98 for the OrganSMNIST Classification Challenge and 0.73 to 0.83 for a brain hemorrhage classification task. Furthermore, pre-training is up to nine times faster due to the dataset reduction.
#### Conclusions:
In conclusion, the proposed approach highlights the importance of dataset quality and provides a transferable approach to improve contrastive pre-training for classification downstream tasks on medical images.

## Code 

### 1) Pre-Training
First, the deep learning model needs to be pre-trained with large datasets of images without annotations. 

Instead of doing the pre-training yourself, you can also download our pre-trained models below and proceed to step 2) Downstream Evaluation.  

##### 1.1) Pre-Train Data Preprocessing
Go to the folder [PreTrain-Data_Preprocessing](https://github.com/Wolfda95/Less_is_More/tree/main/PreTrain-Data_Preprocessing). \
Here we explain where you can download our pre-training datasets and how to convert them from 3D volumes in nifti or dicom format to 2D png images.

##### 1.2) Pre-Train Data Reduction
Go to the folder [PreTrain-Data_Reduction](https://github.com/Wolfda95/Less_is_More/tree/main/PreTrain-Data_Reduction). \
Here is the code to reduce the pre-training dataset using the best performing reduction method "Hash". \
Pre-training with the reduced dataset will improve your downstream results and make your pre-training faster. 

##### 1.2) Pre-Training with Contrastive Learning
Go to the folder [Pre-Training/Contrastive_Learning](https://github.com/Wolfda95/Less_is_More/tree/main/Pre-Training/Contrastive_Learning). \
Here is the code and further explanations for pre-training our model using the contrastive learning method SwAV.  
For other pre-training methods like MoCo or SparK please look at this GitHub Repo: [https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Pre-Training](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Pre-Training)

### 2) Downstream Evaluation
The pre-training is evaluated on three downstream classification tasks. \
You can test the downstream tasks with the pre-trained models you can download below. \
Go to the folder [Downstream_Evaluation](https://github.com/Wolfda95/Less_is_More/tree/main/Downstream_Evaluation) for the the downstream code and further explanations.

## Pre-Trained Models 
You can download the pre-trained model checkpoints here from Google Drive:


| Pre-Train Dataset  | Reduction                    | Model       |Dowwnload Link |
| ------------------ | ---------------------------- |------------ | ------------  |
| PET-CT             | No Reduction                 | ResNet50    |[PET-CT_All](https://drive.google.com/uc?export=download&id=1eBZYl1rXkKJxz42Wu75uzb1kLg8FTv1H)              |
| PET-CT             | Hash-6 Reduced (Best Method) | ResNet50    |[PET-CT_Hash6](https://drive.google.com/uc?export=download&id=11OWRzifq_BXrcFMZ13H0HwS4UGcaiAn_)               |
| LIDC               | No Reduction                 | ResNet50    |[LIDC_All](https://drive.google.com/uc?export=download&id=1hUr_6XdYxjB66ZYEGTqE7b8I88IN9a1l)            | 
| LIDC               | Hash-6 Reduced (Best Method) | ResNet50    |[LIDC_Hash6](https://drive.google.com/uc?export=download&id=1kYFS67jH9s8kAmhNyf5wlRj_Gh9vTK_H)               |


Here is code to initialise a ResNet50 model from PyTorch with the pre-training weights stored in the Checkpoint:  \
(pytorch==1.12.1 torchvision==0.13.1) \
You can also check out the the [Downstream_Evaluation](https://github.com/Wolfda95/Less_is_More/tree/main/Downstream_Evaluation) code where this is already implemented.

```python

pre_train = "SwAV"

# Fill out: 
# Insert the downloaded file hier (.ckpt) 
pre_training_checkpoint = "/path/to/download/model.ckpt"

# PyTorch Resnet Model
res_model = torchvision.models.resnet50()

# Load pre-training weights
state_dict = torch.load(pre_training_checkpoint)

# Match the correct name of the layers between pre-trained model and PyTorch ResNet
# Extraction:
if "module" in state_dict: # (SparK)
    state_dict = state_dict["module"] 
if "state_dict" in state_dict: # (SwAV, MoCo, BYOL) 
    state_dict = state_dict["state_dict"]
# Replacement: 
if pre_train == "SparK" or pre_train == "SwAV":
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}  
elif pre_train == "MoCo":
    state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()} 
elif pre_train == "BYOL":
    state_dict = {k.replace("online_network.encoder.", ""): v for k, v in state_dict.items()}

# Initialisation of the ResNet model with pre-training checkpoints
pretrained_model = res_model.load_state_dict(state_dict, strict=False)

# Check if it works
print(format(pretrained_model))
# If this appears, everything is correct: 
# missing_keys=
  # ['fc.weight', 'fc.bias'] (beacuse the last fully connected layer was not pre-trained) 
# unexpected_keys= 
  # All "projection_head" layers (beacuse SwAV has an aditional projection head for the online clustering) 
```

## Contact
This work was done in a collaboration between the [Clinic of Radiology](https://www.uniklinik-ulm.de/radiologie-diagnostische-und-interventionelle.html) and the [Visual Computing Research Group](https://viscom.uni-ulm.de/) at the Univerity of Ulm.

My Profiles: 
- [Ulm University Profile](https://viscom.uni-ulm.de/members/daniel-wolf/)
- [Personal Website](https://wolfda95.github.io/)
- [Google Scholar](https://scholar.google.de/citations?hl=de&user=vqKsXwgAAAAJ)
- [Orcid](https://orcid.org/0000-0002-8584-5189)
- [LinkedIn](https://www.linkedin.com/in/wolf-daniel/)

If you have any questions, please email me:
[daniel.wolf@uni-ulm.de](mailto:daniel.wolf@uni-ulm.de)

## Cite
```latex
@article{WOLF2024109242,
title = {Less is More: Selective reduction of CT data for self-supervised pre-training of deep learning models with contrastive learning improves downstream classification performance},
author = {Daniel Wolf and Tristan Payer and Catharina Silvia Lisson and Christoph Gerhard Lisson and Meinrad Beer and Michael Götz and Timo Ropinski},
journal = {Computers in Biology and Medicine},
volume = {183},
pages = {109242},
year = {2024},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2024.109242},
}
```
