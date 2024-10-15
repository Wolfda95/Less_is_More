# Pre-Train Data Reduction

Here is the code for reducing the pre-training data with the best-perfoming reduction method Hash-6.

Before running those scripts do the steps in [PreTrain-Data_Preprocessing](https://github.com/Wolfda95/Less_is_More/tree/main/PreTrain-Data_Preprocessing) (Downloading the Pre-Trianing data and converting to 2D png images) 

If you are using Conda on Linux, here is how to get started: 
1. Open your terminal and follow these steps: 
    1. <code>conda create --name Less_is_More_Reduction python==3.10</code>
    2. <code>conda activate Less_is_More_Reduction</code>
    4. <code>cd ...Less_is_More/PreTrain-Data_Reduction</code>
    5. <code>pip install -r requirements.txt</code>
2. Run Hash_LIDC_Volume.py for the LIDC dataset or Hash_PET_CT_Volumepy for the PET-CT dataset and adjust the parser.

If you have your own dataset, use Hash_AllData.py. \
Here you should be able to pass any dataset of 2D png images. 
