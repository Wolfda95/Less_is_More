# Pre-Training with SwAV

We used the implementation of PyTorch Lightning Bolds [https://lightning.ai/docs/pytorch/stable/ecosystem/bolts.html](https://lightning.ai/docs/pytorch/stable/ecosystem/bolts.html)

### How to Start:
1. Download the LIDC and/or the PET-CT data and run the preprocessing script as explained here: [https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Pre-Training/Data_Preprocessing](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Pre-Training/Data_Preprocessing)

#### Option 1: Use the latest PyTorch Lightning Bolts implementation
You can use the implementation of PyTorch Lightning Bolts. You only have to change the data loading. 
- SwAV: [https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/models/self_supervised/swav/swav_module.py](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/models/self_supervised/swav/swav_module.py)

#### Option 2: Use our PyTorch Lightning Bolts adapion
2. Change the folder structure of the preprocessed data so that you have one subfolder were all the images are stored in: 
    ```bash
        LIDC-Data
         /        
       train       
    ```
2. Open your terminal and follow these steps: 
    1. <code>conda create --name Less_is_More_Contrastive python==3.10</code>
    2. <code>conda activate Less_is_More_Contrastive</code>
    3. <code>conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia</code>
    4. <code>cd .../Less_is_More/Pre-Training/Contrastive_Learning/</code>
    5. <code>pip install -r requirements.txt</code>
4. Start the pre-training with a bash script: \
   SwAV: 
    ```bash
    #!/bin/bash

    wandb login your_login_id
    python .../SSL-MedicalImagining-CL-MAE/Pre-Training/Contrastive_Learning/pl_bolts/models/self_supervised/swav/swav_module_lidc.py \
    --save_path /path/where/results/should/be/saved \
    --data_dir /path/to/the/LIDC-Data \
    --model Some_Name_for_WandB \
    --test Some_Name_for_WandB \
    --project WandB_project_name \
    --batch_size 128 \
    --group Bs_128 \
    --tags ["500Proto_Color2x04-2x02-Blur-Crop"] \
    --learning_rate 0.15 \
    --final_lr 0.00015 \
    --start_lr 0.3 \
    --freeze_prototypes_epochs 313 \
    --accumulate_grad_batches 1 \
    --optimizer lars \
    ```

For further information and other setting please refere to the PyTorch Lightning Bolds github: [https://github.com/Lightning-Universe/lightning-bolts/tree/master](https://github.com/Lightning-Universe/lightning-bolts/tree/master)
