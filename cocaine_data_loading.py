#Transformer model 
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd 
import os
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

seed = 2351564

def load_COCAINE_data(batch_size = 32):

    subjects_tsv = pd.read_csv('COCAINE/participants.tsv', sep='\t')

    #Retrieve BOLD signals from fmri data
    BOLD_signals = {}
    avg_BOLD_signals = {}
    sex = []
    labels = []

    for _, row in tqdm(subjects_tsv.iterrows()):
        subject = row['participant_id']
        path = 'COCAINE/'+subject+'/BOLD_time_series.mat'
        try: 
            group = row['group']
            if not np.isnan(group):
                BOLD_signals[subject] = scipy.io.loadmat(path)
                labels.append(row['group'])
                sex.append(row['sex'])
            else :
                continue
        except:
            continue

    keys = list(BOLD_signals.keys())

    for id in keys :
        BOLD_signals[id] = BOLD_signals[id]['all_time_series'][0]

    numerical_labels = LabelEncoder().fit_transform(labels)

    labels_tensor = torch.tensor(numerical_labels)
    #sex_tensor = torch.tensor(sex)

    error_voxels = []

    #Formate to tensor stucture
    for id in tqdm(keys):
        a = np.empty((166, 196))
        for voxel in range(166):
            mean_signal = np.mean(BOLD_signals[id][voxel], axis=-1)
            try : 
                a[voxel] = mean_signal
            except:
                error_voxels.append((id, voxel))
                continue
        avg_BOLD_signals[id] = a
    #Now, avg_BOLD_signals is a dictionary containing the average BOLD signals 
    # for each subject at each voxel (each row represents a voxel)


    #Formate to tensor stucture
    subjects = list(avg_BOLD_signals.keys())
    data_list = list(avg_BOLD_signals.values())
    data_array = np.stack(data_list, axis=0)

    #Normalize data
    means = np.mean(data_array, axis=(0, 2), keepdims=True)
    stds = np.std(data_array, axis=(0, 2), keepdims=True)
    data_normalized = (data_array - means) / (stds + 1e-8) #1e-8 is added to avoid dividing by zero 
    data_normalized = np.swapaxes(data_normalized, 1, 2) #So that the time dimension remains the same and 116-length vectors are presented as input to the model


    train_data, test_data, train_labels, test_labels = train_test_split(data_normalized, labels_tensor, test_size=0.2, random_state=seed, stratify=labels_tensor)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=seed, stratify=train_labels)

    train_loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float32), train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_data, dtype=torch.float32), val_labels), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_data, dtype=torch.float32), test_labels), batch_size=batch_size, shuffle=True)

    #Creating the objects 
    globals()['BOLD_signals'] = data_array[:,:,10:] #Remove the 10 first timesteps 
    globals()['labels'] = labels_tensor
    globals()['train_loader'] = train_loader
    globals()['val_loader'] = val_loader
    globals()['test_loader'] = test_loader
    globals()['train_data'] = train_data
    globals()['val_data'] = val_data
    globals()['test_data'] = test_data
    globals()['train_labels'] = train_labels
    globals()['val_labels'] = val_labels
    globals()['test_labels'] = test_labels
    globals()['batch_size'] = batch_size
    globals()['subjects'] = subjects
    globals()['subjects_tsv'] = subjects_tsv



load_COCAINE_data()





