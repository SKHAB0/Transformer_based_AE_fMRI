# Transformer based AutoEncoders on rs-fMRI data
This Msc. project constitute the main part of my thesis at Imperial College London (2022-2023). This project has been supervised by A. Luati. 

To use this repo, make sure that Python and Matlab are installed on your machine. 
Also, you will have to download SPM12, a fMRI data preprocessing Toolbox : https://www.fil.ion.ucl.ac.uk/spm/software/spm12/. 

The packages can be installed via pip using the following command line: 

```bash
pip install -r requirements.txt
```

## Download the data

To download the data and unzip data for each phenomenon (e.g. DEPRESSION) execute the following command : 

```bash
bash download_depression_data.sh
```

```matlab
find DEPRESSION -type f -name "*.gz" -exec gunzip -f {} \;
```

# Preprocess the data

For each data folder (e.g. DEPRESSION), you can access three MATLAB scipts using SPM12 to preprocess anatomical and functional images and to extract BOLD signal. 
Type the following command line : 

```bash
matlab -r RunPreproc_1stLevel_job_func_DEPRESSION
matlab -r RunPreproc_1stLevel_job_anat_DEPRESSION
matlab -r BOLD_extraction_DEPRESSION
```

For each dataset, EDA_***.py contains the exploratory data analysis, metrics and plots that have been shared on the Msc. thesis report. 

The file model_and_layers.py contains core modules that have been used to build the three different encoder heads : Autocorrelation head using the Autoformer model, Decomposition head using Attention heads and a Decomposition layer, and a standard implementation of Attention transformers. 

The python files names ***_study.py are related to the dataset ***, and are defining for each dataset the physical loss, training loop, bayesian optimization loop and models, that have been produced alongside this study. 
