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

torch.manual_seed(2351564) #CID

#Parallelizing 
num_threads = torch.get_num_threads()
# Set the number of threads to the maximum available
torch.set_num_threads(num_threads)

#Hyperparameters for the depression dataset: 
embed_dim = 166
time_length = 196
batch_size = 32

#Execute data loading in Pytorch loaders 
filename = 'bilingualism_data_loading.py'
with open(filename, 'r') as file:
    script_content = file.read()
exec(script_content)

# Create boolean masks for each label
mask_MC = labels == 0
mask_LB = labels == 1
mask_EB = labels == 2

# Create tensors for each label by indexing with the boolean masks
BOLD_MC = BOLD_signals[mask_MC]
BOLD_LB = BOLD_signals[mask_LB]
BOLD_EB = BOLD_signals[mask_EB]

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

import hashlib

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

mean_bold_MC = np.mean(BOLD_MC, axis = 0).T
mean_bold_LB = np.mean(BOLD_LB, axis = 0).T
mean_bold_EB = np.mean(BOLD_EB, axis = 0).T

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

R_bold_MC = pearson_correlation_matrix(mean_bold_MC)
R_bold_LB = pearson_correlation_matrix(mean_bold_LB)
R_bold_EB = pearson_correlation_matrix(mean_bold_EB)

np.fill_diagonal(R_bold_MC, 0)
np.fill_diagonal(R_bold_LB, 0)
np.fill_diagonal(R_bold_EB, 0)

Z_bold_MC = fisher_z_transformation_matrix(R_bold_MC)
Z_bold_LB = fisher_z_transformation_matrix(R_bold_LB)
Z_bold_EB = fisher_z_transformation_matrix(R_bold_EB)



from matplotlib.colors import LinearSegmentedColormap

c_dict = {'red':   [(0.0,  0.0, 0.0),  # Bleu pour les valeurs < 0
                    (0.5,  0.99, 0.99),  # Blanc pour la valeur 0
                    (1.0,  180/255, 180/255)],  # Rouge pour les valeurs > 0

          'green': [(0.0,  0.0, 0.0),  # Bleu
                    (0.5,  0.99, 0.99),  # Blanc
                    (1.0,  0.0, 0.0)],  # Rouge

          'blue':  [(0.0,  180/255, 180/255),  # Bleu
                    (0.5,  0.99, 0.99),  # Blanc
                    (1.0,  0.0, 0.0)]  # Rouge
        }

# Création de la colormap personnalisée
custom_map = LinearSegmentedColormap('BlueWhiteRed', c_dict)

# colors = [(0, 0,180/255),
#           (0.99, 0.99, 0.99), 
#           (180/255, 0,0)]  # Grey to Red
# n_bins = 16 # Discretizes the colormap into these many bins
# cmap_name = "GrRd"
# custom_map = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


Z_bold_EB_filtered = np.where(abs(Z_bold_EB) > 0.65, Z_bold_EB, 0)
Z_bold_LB_filtered = np.where(abs(Z_bold_LB) > 0.65, Z_bold_LB, 0)
Z_bold_MC_filtered = np.where(abs(Z_bold_MC) > 0.65, Z_bold_MC, 0)

node_angles = circular_layout(
    new_order, new_order, start_pos=90, group_boundaries=[0, len(rois['nom_l']) / 2])

fig, ax = plt.subplots(
    figsize=(20, 20), facecolor="white", subplot_kw=dict(projection="polar"))

plot_connectivity_circle(
    Z_bold_EB_filtered,
    rois['nom_l'],
    #n_lines= (np.abs(Z_bold_depressed)>0.65).sum(),
    node_angles=node_angles,
    node_colors=rois['color'],
    title="Connectogram of the EB group",
    ax=ax,
    facecolor='none',
    textcolor='black',
    fontsize_title=20,
    fontsize_names=12,
    #padding=6,
    #colorbar_size=0.2,
    #vmin=-0.5,
    #vmax=0.5,
    colormap=custom_map
)

#save figure 
fig.savefig('figures/bilingualism/connectogram_EB.png', 
            facecolor=fig.get_facecolor(), 
            edgecolor='none', 
            bbox_inches='tight')


node_angles = circular_layout(
    new_order, new_order, start_pos=90, group_boundaries=[0, len(rois['nom_l']) / 2])

fig, ax = plt.subplots(
    figsize=(20, 20), facecolor="white", subplot_kw=dict(projection="polar"))

plot_connectivity_circle(
    Z_bold_LB_filtered,
    rois['nom_l'],
    #n_lines= (np.abs(Z_bold_depressed)>0.65).sum(),
    node_angles=node_angles,
    node_colors=rois['color'],
    title="Connectogram of the LB group",
    ax=ax,
    facecolor='none',
    textcolor='black',
    fontsize_title=20,
    fontsize_names=12,
    #padding=6,
    #colorbar_size=0.2,
    #vmin=-0.5,
    #vmax=0.5,
    colormap=custom_map
)

#save figure 
fig.savefig('figures/bilingualism/connectogram_LB.png', 
            facecolor=fig.get_facecolor(), 
            edgecolor='none', 
            bbox_inches='tight')



node_angles = circular_layout(
    new_order, new_order, start_pos=90, group_boundaries=[0, len(rois['nom_l']) / 2])

fig, ax = plt.subplots(
    figsize=(20, 20), facecolor="white", subplot_kw=dict(projection="polar"))

plot_connectivity_circle(
    Z_bold_MC_filtered,
    rois['nom_l'],
    #n_lines= (np.abs(Z_bold_depressed)>0.65).sum(),
    node_angles=node_angles,
    node_colors=rois['color'],
    title="Connectogram of the MC group",
    ax=ax,
    facecolor='none',
    textcolor='black',
    fontsize_title=20,
    fontsize_names=12,
    #padding=6,
    #colorbar_size=0.2,
    #vmin=-0.5,
    #vmax=0.5,
    colormap=custom_map
)

#save figure 
fig.savefig('figures/bilingualism/connectogram_MC.png', 
            facecolor=fig.get_facecolor(), 
            edgecolor='none', 
            bbox_inches='tight')


