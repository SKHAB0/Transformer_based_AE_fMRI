#Exploratory data analysis - depression dataset

#Transformer model 
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd 
import os
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt 
import math
import torch.optim as optim
from nilearn import plotting
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle
import hashlib
from matplotlib.colors import LinearSegmentedColormap


#Load data
subjects_tsv = pd.read_csv('DEPRESSION/participants.tsv', sep='\t')

#Hyperparameters for the depression dataset: 
embed_dim = 166
time_length = 196
batch_size = 32

#Execute data loading
filename = 'depression_data_loading.py'
with open(filename, 'r') as file:
    script_content = file.read()
exec(script_content)

#Splitting the data into depressed and healthy

#Male are 0, and female are 1

is_men_tensor = torch.tensor(is_men)

depressed_female = labels*(~is_men_tensor)
depressed_male = labels*is_men_tensor
healthy_female = ~labels*(~is_men_tensor)
healthy_male = ~labels*is_men_tensor


BOLD_signals_depressed = BOLD_signals[labels]
BOLD_signals_healthy = BOLD_signals[~labels]

#Splitting men and women 
BOLD_signals_depressed_male = BOLD_signals[depressed_male]
BOLD_signals_depressed_female = BOLD_signals[depressed_female]
BOLD_signals_healthy_male = BOLD_signals[healthy_male]
BOLD_signals_healthy_female = BOLD_signals[healthy_female]

#Retrieve atlas information

rois = pd.read_table('AAL3/ROI_MNI_V7_vol.txt', sep = '\t')


#Reorder to match the order of the atlas
n = rois.shape[0]
even_indices = list(range(0, n, 2))
odd_indices = list(range(1, n, 2))
new_order = even_indices + odd_indices[::-1]

rois = rois.iloc[new_order]
rois.reset_index(drop=True, inplace=True)

extract_group = lambda x: x[:3] if len(x) >= 3 else x
rois['Group'] = rois['nom_l'].apply(extract_group)

#Create colours for regions 

def string_to_distinct_rgb(s):
    if len(s) != 3:
        raise ValueError("The string must contain exactly 3 letters.")
    # Generate a hash of the string
    m = hashlib.md5()
    m.update(s.encode('utf-8'))
    digest = m.digest()
    
    # Use the first 3 bytes of the hash as RGB values
    return tuple(digest[i]/255 for i in range(3))


rois['color'] = rois['Group'].apply(string_to_distinct_rgb)


#Compute matrices 

mean_bold_depressed = np.mean(BOLD_signals_depressed, axis = 0).T
mean_bold_healthy = np.mean(BOLD_signals_healthy, axis = 0).T

mean_bold_depressed_male = np.mean(BOLD_signals_depressed_male, axis = 0).T
mean_bold_depressed_female = np.mean(BOLD_signals_depressed_female, axis = 0).T
mean_bold_healthy_male = np.mean(BOLD_signals_healthy_male, axis = 0).T
mean_bold_healthy_female = np.mean(BOLD_signals_healthy_female, axis = 0).T


def pearson_correlation_matrix(X):
    # Standardize the columns
    Z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # Calculate the Pearson correlation matrix
    R = np.matmul(Z.T,Z) / (X.shape[0])
    return R

def fisher_z_transformation_matrix(R):
    # Apply Fisher's Z-transformation to each element of the matrix
    Z = 0.5 * np.log((1 + R) / (1 - R))
    return Z

R_bold_depressed = pearson_correlation_matrix(mean_bold_depressed)
R_bold_healthy = pearson_correlation_matrix(mean_bold_healthy)
R_bold_depressed_male = pearson_correlation_matrix(mean_bold_depressed_male)
R_bold_depressed_female = pearson_correlation_matrix(mean_bold_depressed_female)
R_bold_healthy_male = pearson_correlation_matrix(mean_bold_healthy_male)
R_bold_healthy_female = pearson_correlation_matrix(mean_bold_healthy_female)

np.fill_diagonal(R_bold_depressed, 0)
np.fill_diagonal(R_bold_healthy, 0)
np.fill_diagonal(R_bold_depressed_male, 0)
np.fill_diagonal(R_bold_depressed_female, 0)
np.fill_diagonal(R_bold_healthy_male, 0)
np.fill_diagonal(R_bold_healthy_female, 0)


Z_bold_depressed = fisher_z_transformation_matrix(R_bold_depressed)
Z_bold_healthy = fisher_z_transformation_matrix(R_bold_healthy)
Z_bold_depressed_male = fisher_z_transformation_matrix(R_bold_depressed_male)
Z_bold_depressed_female = fisher_z_transformation_matrix(R_bold_depressed_female)
Z_bold_healthy_male = fisher_z_transformation_matrix(R_bold_healthy_male)
Z_bold_healthy_female = fisher_z_transformation_matrix(R_bold_healthy_female)

#Create scale 

colors = [(0, 0,180/255),(0.99, 0.99, 0.99), (180/255, 0,0)]  # Grey to Red
n_bins = 16 # Discretizes the colormap into these many bins
cmap_name = "GrRd"
custom_map = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

Z_bold_depressed_filtered = np.where(np.abs(Z_bold_depressed) > 0.65, Z_bold_depressed, 0)
Z_bold_healthy_filtered = np.where(np.abs(Z_bold_healthy) > 0.65, Z_bold_healthy, 0)

Z_bold_depressed_filtered_male = np.where(np.abs(Z_bold_depressed_male) > 0.65, Z_bold_depressed_male, 0)
Z_bold_depressed_filtered_female = np.where(np.abs(Z_bold_depressed_female) > 0.65, Z_bold_depressed_female, 0)
Z_bold_healthy_filtered_male = np.where(np.abs(Z_bold_healthy_male) > 0.65, Z_bold_healthy_male, 0)
Z_bold_healthy_filtered_female = np.where(np.abs(Z_bold_healthy_female) > 0.65, Z_bold_healthy_female, 0)


#Plots

##DEPRESSED MALE 
node_angles = circular_layout(
    new_order, new_order, start_pos=90, group_boundaries=[0, len(rois['nom_l']) / 2])

fig, ax = plt.subplots(
    figsize=(20, 20), facecolor="white", subplot_kw=dict(projection="polar"))

plot_connectivity_circle(
    Z_bold_depressed_filtered_male,
    rois['nom_l'],
    #n_lines= (np.abs(Z_bold_depressed)>0.65).sum(),
    node_angles=node_angles,
    node_colors=rois['color'],
    title="Connectogram of the depressed group (Male)",
    ax=ax,
    facecolor='none',
    textcolor='black',
    fontsize_title=20,
    fontsize_names=12,
    colormap=custom_map
)

#save figure 
fig.savefig('figures/connectogram_depressed_male.png', 
            facecolor=fig.get_facecolor(), 
            edgecolor='none', 
            bbox_inches='tight')


##DEPRESSED FEMALE
node_angles = circular_layout(
    new_order, new_order, start_pos=90, group_boundaries=[0, len(rois['nom_l']) / 2])

fig, ax = plt.subplots(
    figsize=(20, 20), facecolor="white", subplot_kw=dict(projection="polar"))

plot_connectivity_circle(
    Z_bold_depressed_filtered_female,
    rois['nom_l'],
    node_angles=node_angles,
    node_colors=rois['color'],
    title="Connectogram of the depressed group (Female)",
    ax=ax,
    facecolor='none',
    textcolor='black',
    fontsize_title=20,
    fontsize_names=12,
    colormap=custom_map,

)

#save figure 
fig.savefig('figures/connectogram_depressed_female.png', 
            facecolor=fig.get_facecolor(), 
            edgecolor='none', 
            bbox_inches='tight')


##HEALTHY MALE
node_angles = circular_layout(
    new_order, new_order, start_pos=90, group_boundaries=[0, len(rois['nom_l']) / 2])

fig, ax = plt.subplots(
    figsize=(20, 20), facecolor="white", subplot_kw=dict(projection="polar"))

plot_connectivity_circle(
    Z_bold_healthy_filtered_male,
    rois['nom_l'],
    node_angles=node_angles,
    node_colors=rois['color'],
    title="Connectogram of the healthy group (Male)",
    ax=ax,
    facecolor='none',
    textcolor='black',
    fontsize_title=20,
    fontsize_names=12,
    colormap=custom_map,

)

#save figure 
fig.savefig('figures/connectogram_healthy_male.png', 
            facecolor=fig.get_facecolor(), 
            edgecolor='none', 
            bbox_inches='tight')


##HEALTHY FEMALE 

node_angles = circular_layout(
    new_order, new_order, start_pos=90, group_boundaries=[0, len(rois['nom_l']) / 2])

fig, ax = plt.subplots(
    figsize=(20, 20), facecolor="white", subplot_kw=dict(projection="polar"))

plot_connectivity_circle(
    Z_bold_healthy_filtered_female,
    rois['nom_l'],
    node_angles=node_angles,
    node_colors=rois['color'],
    title="Connectogram of the healthy group (Female)",
    ax=ax,
    facecolor='none',
    textcolor='black',
    fontsize_title=20,
    fontsize_names=12,
    colormap=custom_map,

)

#save figure 
fig.savefig('figures/connectogram_healthy_female.png', 
            facecolor=fig.get_facecolor(), 
            edgecolor='none', 
            bbox_inches='tight')




##Longitudinal study for subject 23546

sub = 'sub-23546'
matching_indices = [index for index, element in enumerate(subjects) if element.startswith(sub)]

BOLD_23546 = BOLD_signals[matching_indices]

R_23546 = {i : pearson_correlation_matrix(np.mean(BOLD_23546[i:i+4], axis = 0).T) for i in range(0, len(BOLD_23546), 4)}
Z_23546 = {key : fisher_z_transformation_matrix(value) for key, value in R_23546.items()}

for i,x in Z_23546.items():

    x_filtered = np.where(np.abs(x) > 0.65, x, 0)

    session = subjects[matching_indices[i]].replace('_echo-1', '')

    node_angles = circular_layout(
    new_order, new_order, start_pos=90, group_boundaries=[0, len(rois['nom_l']) / 2])

    fig, ax = plt.subplots(
        figsize=(20, 20), facecolor="white", subplot_kw=dict(projection="polar"))

    plot_connectivity_circle(
        x_filtered,
        rois['nom_l'],
        node_angles=node_angles,
        node_colors=rois['color'],
        title="Connectogram of "+ session,
        ax=ax,
        facecolor='none',
        textcolor='black',
        fontsize_title=20,
        fontsize_names=12,
        colormap=custom_map
        )

    fig.savefig('figures/connectogram_' + session + '.png', 
        facecolor=fig.get_facecolor(), 
        edgecolor='none', 
        bbox_inches='tight')
    

