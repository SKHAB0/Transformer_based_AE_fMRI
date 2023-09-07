
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
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
import numpy as np
from joblib import Parallel, delayed
import torch
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from models_and_layers import *
from collections import defaultdict

seed = 2351564

torch.manual_seed(seed) #CID

#Parallelizing 
num_threads = torch.get_num_threads()
torch.set_num_threads(num_threads)

#Execute data loading in Pytorch loaders 
filename = 'depression_data_loading.py'
with open(filename, 'r') as file:
    script_content = file.read()
exec(script_content)

#Hyperparameters for the depression dataset: 
embed_dim = BOLD_signals.shape[1]
time_length = BOLD_signals.shape[2]
batch_size = 32



def physical_loss(inputs, a, b, c, K):
    # Calculate the first derivative
    first_derivative = (inputs[:, 1:, :] - inputs[:, :-1, :])

    # Calculate the second derivative
    second_derivative = (first_derivative[:, 1:, :] - first_derivative[:, :-1, :])
    #print('Here 1')

    # Trim inputs to match the dimensions of the second derivative
    trimmed_inputs = inputs[:, :-2, :]
    #print('Here 2')
    # Apply the physical equation
    phys_eq = a * second_derivative + b * first_derivative[:, :-1, :] + c * trimmed_inputs + K
    #print('Here 3')
    # Square the result
    phys_loss = torch.sum(phys_eq ** 2)
    #print('Here 4')

    return phys_loss/inputs.shape[2]

def training(model, 
             train_loader=train_loader, 
             val_loader=val_loader, 
             criterion=nn.MSELoss(), 
             num_epochs=100, 
             patience=10, 
             lr=1e-3, 
             physical_position='latent', 
             loss_function='physical', 
             lambda_ = 0.1):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # If you have more than one GPU, wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Let's gooooooooo : {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device) #Send the model to GPU or CPU

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=lr)  # Added model.parameters() to the optimizer

    no_improvement_count = 0
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    last_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch[0]
            outputs, latent_representation = model(inputs, mask = None)

            # Original loss
            loss = criterion(outputs, inputs)

            if physical_position == 'end':
                # Physical loss
                phys_loss = physical_loss(outputs, model.a, model.b, model.c, model.K)
            else:
                phys_loss = physical_loss(latent_representation, model.a, model.b, model.c, model.K)

            if loss_function == 'physical':
            # Total loss
                total_loss = (loss + lambda_*phys_loss)
            else :
                total_loss = loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += total_loss.item() * inputs.size(0)

        # Validation Loop (optional)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0]
                outputs, latent_representation = model(inputs, None)
                loss = criterion(outputs, inputs)
                if physical_position == 'end':
                    # Physical loss
                    phys_loss = physical_loss(outputs, model.a, model.b, model.c, model.K)
                else:
                    phys_loss = physical_loss(latent_representation, model.a, model.b, model.c, model.K)
                if loss_function == 'physical':
                    # Total loss
                    val_loss += (loss + (lambda_*phys_loss)).item() * inputs.size(0)
                else:
                    val_loss += loss.item() * inputs.size(0)
                
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        last_epoch = epoch

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {np.round(train_loss, 5)}, Validation Loss: {np.round(val_loss, 5)}")  # Fixed the print statement

        # Early stopping and best model saving logic
        if val_loss < (best_val_loss - 1e-4):
            best_val_loss = val_loss
            no_improvement_count = 0
            best_model = model
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            no_improvement_count += 1
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print("\n")
            print(f"Last epoch {epoch + 1}/{num_epochs}, Training Loss: {np.round(train_loss, 5)}, Validation Loss: {np.round(val_loss, 5)}")
            print("Early stopping triggered. Stopping training.")
            break

    return best_model, train_losses, val_losses, last_epoch, optimizer

def train_model(embed_dim=embed_dim, 
                pattern=[32, 16, 4, 2], 
                heads=2, 
                position='latent', 
                dropout=0.5, 
                forward_expansion=2, 
                lr=1e-3, 
                loss_function = 'physical', 
                lambda_ = 0.1,
                model_type = 'Autocorr_Transformer'):
    
    layers = '_'.join(map(str, pattern))

    if model_type == 'Autocorr_Transformer':
        model = Autocorr_TransformerAutoencoder(input_dim=embed_dim, 
                                            intermediate_dims=pattern, 
                                            heads=heads, 
                                            dropout=dropout, 
                                            forward_expansion=forward_expansion)
    elif model_type == 'Decomposition_Transformer':
        model = Decomposition_Transformer_Autoencoder(input_dim=embed_dim,
                                                        intermediate_dims=pattern,
                                                        heads=heads,
                                                        dropout=dropout,
                                                        forward_expansion=forward_expansion)
    elif model_type == 'Standard_Transformer':
        model = Standard_Transformer_Autoencoder(input_dim=embed_dim,
                                                    intermediate_dims=pattern,
                                                    heads=heads,
                                                    dropout=dropout,
                                                    forward_expansion=forward_expansion)

    
    trained_model, train_losses, val_losses, last_epoch, opt = training(model, 
                                                                        num_epochs=400, 
                                                                        patience=10, 
                                                                        lr=lr, 
                                                                        physical_position=position, 
                                                                        loss_function=loss_function, 
                                                                        lambda_ = lambda_)
    save_info = {
        'model': trained_model,
        'epoch': last_epoch,  # Number of epochs trained
        'model_state_dict': trained_model.state_dict(),  # Model parameters
        'optimizer_state_dict': opt.state_dict(),  # Optimizer parameters
        'train_loss': train_losses,  # Training loss
        'val_loss': val_losses # Validation loss
    }



    new_path = 'trained_models_depression/' + model_type + '/' + position + '/' + layers
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    torch.save(save_info, new_path + '/trained_model_' + str(heads) + 'heads' + '_FE' + str(forward_expansion) + '_lambda-' + str(np.round(lambda_,4)) + '_'+ loss_function + '_'+ model_type +'.pth')
    
    return save_info

def metrics(model):
    output_train = model(train_loader.dataset.tensors[0], None)[0]
    output_train = output_train.detach().numpy()

    output_val = model(val_loader.dataset.tensors[0], None)[0]
    output_val = output_val.detach().numpy()
    output_test = model(test_loader.dataset.tensors[0], None)[0]
    output_test = output_test.detach().numpy()

    mse_train = np.mean((output_train-train_data)**2)
    mse_val = np.mean((output_val-val_data)**2)
    mse_test = np.mean((output_test-test_data)**2)
    return mse_train, mse_val, mse_test

#Search for the best model 

def objective(first_layer, second_layer, third_layer, latent, lambda_):
    # Map the continuous variables to the discrete grid
    first_layer_options = [32, 64, 128]
    second_layer_options = [16, 32, 64]
    third_layer_options = [4, 8, 16]
    latent_options = [2, 4]
    
    first_layer = first_layer_options[int(first_layer)]
    second_layer = second_layer_options[int(second_layer)]
    third_layer = third_layer_options[int(third_layer)]
    latent = latent_options[int(latent)]
    
    params = {
        'embed_dim': embed_dim,  # You can also make this a variable if needed
        'pattern': [first_layer, second_layer, third_layer, latent],  # Must convert to integers
        'heads': 2,  # This can be a variable too
        'position': 'latent',
        'dropout': 0.5,
        'forward_expansion': 2,
        'lr': 1e-3,
        'loss_function': 'physical',
        'lambda_': lambda_
    }
    
    # Train the model
    save_info = train_model(**params)
    
    # Get the metrics
    trained_model = save_info['model']
    _, mse_val, _ = metrics(trained_model)
    
    # Return the negative of MSE to maximize the objective function
    return -mse_val

# Use Bayesian Optimization to search for the best hyperparameters
pbounds = {
    'first_layer': (0, 2),  # 3 options: index 0, 1, or 2 from the first_layer_options list
    'second_layer': (0, 2),  # 3 options
    'third_layer': (0, 2),  # 3 options
    'latent': (0, 1),  # 2 options
    'lambda_': (0.01, 1)
}

# optimizer = BayesianOptimization(
#     f=objective,
#     pbounds=pbounds,
#     verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#     random_state=seed,
# )

# optimizer.maximize(
#     init_points=5,  # Number of random points to start with
#     n_iter=30,  # Number of iterations to perform
#     verbose = 1
# )


#Running multiple seeds at the same time to test the maximum number of models

# def run_single_optimization(seed):
#     optimizer = BayesianOptimization(
#         f=objective,
#         pbounds=pbounds,
#         verbose=2,
#         random_state=seed,
#     )
#     optimizer.maximize(
#         init_points=5,
#         n_iter=10
#     )
#     return optimizer.max

# best_results = Parallel(n_jobs=num_threads)(delayed(run_single_optimization)(seed) for seed in range(num_threads))





#It turns out that the best model is : 

# result_dict = defaultdict(dict)

# for root, dirs, files in tqdm(os.walk('trained_models_depression/latent')):
#     for file in files:
#         if file.endswith('.pth'):
            
#             model_path = os.path.join(root, file)
#             model = torch.load(model_path)
            
#             try :
#             # Apply metrics function
#                 values = metrics(model['model'])
#             except:
#                 continue
            
#             # Store the metrics in the dictionary under the appropriate model
#             result_dict[root+'/'+file] = values


# df = pd.DataFrame(result_dict).T
# df.columns = ['mse_train', 'mse_val', 'mse_test']

# df['lambda'] = df.index.str.split('_').str[-2].str.split('-').str[-1]
# df['lambda'] = df['lambda'].astype(float).round(5)

# df['pattern'] = df.index.str.split('/').str[2].str.replace('_', ' ')

# df = df[['pattern', 'lambda', 'mse_train', 'mse_val', 'mse_test']]


#Comparing models 
train_model(model_type = 'Decomposition_Transformer', heads = 1)
train_model(model_type = 'Standard_Transformer', heads = 1)
train_model(model_type = 'Autocorr_Transformer', heads = 1)


model_autocorr = torch.load('trained_models_depression/Autocorr_Transformer/latent/32_16_4_2/trained_model_2heads_FE2_lambda-0.1_physical_Autocorr_Transformer.pth')
model_decomposition = torch.load('trained_models_depression/Decomposition_Transformer/latent/32_16_4_2/trained_model_1heads_FE2_lambda-0.1_physical_Decomposition_Transformer.pth')
model_standard = torch.load('trained_models_depression/Standard_Transformer/latent/32_16_4_2/trained_model_2heads_FE2_lambda-0.1_physical_Standard_Transformer.pth')

# a = metrics(model_autocorr['model'])
# b = metrics(model_decomposition['model'])
# c = metrics(model_standard['model'])

# df = pd.DataFrame([a,b,c], columns=['mse_train', 'mse_val', 'mse_test'])
# print(df.to_latex(index = False))

# df['epochs'] = [model_autocorr['epoch'], model_decomposition['epoch'], model_standard['epoch']]

# print(df.to_latex(index = True))


#################### PLOTS ##############################

# plot_df_autocorr = pd.DataFrame({'MSE train': model_autocorr['train_loss'],
#                                 'MSE val': model_autocorr['val_loss']})

# plot_df_decomposition = pd.DataFrame({'MSE train': model_decomposition['train_loss'],
#                                 'MSE val': model_decomposition['val_loss']})

# plot_df_standard = pd.DataFrame({'MSE train': model_standard['train_loss'],
#                                 'MSE val': model_standard['val_loss']})

# #Line plots for each head 

# fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# plot_df_autocorr.plot(ax=ax[0])
# ax[0].set_title('Autocorr Transformer')
# ax[0].set_xlabel('Epoch')
# ax[0].set_ylabel('MSE')

# plot_df_decomposition.plot(ax=ax[1])
# ax[1].set_title('Decomposition Transformer')
# ax[1].set_xlabel('Epoch')
# ax[1].set_ylabel('MSE')

# plot_df_standard.plot(ax=ax[2])
# ax[2].set_title('Standard Transformer')
# ax[2].set_xlabel('Epoch')
# ax[2].set_ylabel('MSE')

# plt.tight_layout()
# plt.savefig('figures/depression/losses.png')


#Encoding the data 
encoded_train = model_decomposition['model'].encode(train_loader.dataset.tensors[0], None).detach().numpy()
labels_train = train_loader.dataset.tensors[1].detach().numpy()

encoded_val = model_decomposition['model'].encode(val_loader.dataset.tensors[0], None).detach().numpy()
labels_val = val_loader.dataset.tensors[1].detach().numpy()

encoded_test = model_decomposition['model'].encode(test_loader.dataset.tensors[0], None).detach().numpy()
labels_test = test_loader.dataset.tensors[1].detach().numpy()


#Splitting into depressed and non-depressed



depressed_train = encoded_train[labels_train == 1]
healthy_train = encoded_train[labels_train == 0]

shape_depressed = depressed_train.shape
shape_healthy = healthy_train.shape


########################## HMM ##############################

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import time 

scaler = StandardScaler()

scaled_depressed_train = scaler.fit_transform(depressed_train.reshape(-1, 2))
scaled_healthy_train = scaler.fit_transform(healthy_train.reshape(-1, 2))


# Initialize and train the HMM for the 'depressed' class
lengths_depressed = [shape_depressed[1]] * shape_depressed[0]  # 868 sequences each of length 196
depressed_hmm = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000000, tol=1e-4)

lengths_healthy = [shape_healthy[1]] * shape_healthy[0]  # 868 sequences each of length 196
healthy_hmm = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000000, tol=1e-4)

start = time.time()
depressed_hmm.fit(scaled_depressed_train, lengths=lengths_depressed)
end = time.time()
print('Time for trainig HMM depressed: ')
print(end - start)


# Initialize and train the HMM for the 'healthy' class
start = time.time()
healthy_hmm.fit(scaled_healthy_train, lengths=lengths_healthy)
end = time.time()
print('Time for trainig HMM depressed: ')
print(end - start)


depressed_scores = []
healthy_scores = []


for i in tqdm(range(encoded_val.shape[0])):
    sample = encoded_val[i]#.reshape(-1, 196)  # Reshape to fit HMM input format
    actual_label = labels_val[i]
    
    depressed_scores.append(depressed_hmm.score(sample))
    healthy_scores.append(healthy_hmm.score(sample))


for i in tqdm(range(encoded_test.shape[0])):
    sample = encoded_test[i]#.reshape(-1, 196)  # Reshape to fit HMM input format
    actual_label = labels_test[i]
    
    depressed_scores.append(depressed_hmm.score(sample))
    healthy_scores.append(healthy_hmm.score(sample))


mean_depressed = np.mean(depressed_scores)
std_depressed = np.std(depressed_scores)

mean_healthy = np.mean(healthy_scores)
std_healthy = np.std(healthy_scores)

z_depressed_scores = (depressed_scores - mean_depressed) / std_depressed
z_healthy_scores = (healthy_scores - mean_healthy) / std_healthy


predictions = z_depressed_scores > z_healthy_scores

val_test_labels = np.concatenate((labels_val, labels_test))

TP = np.sum(predictions & val_test_labels)
TN = np.sum(~predictions & ~val_test_labels)
FP = np.sum(predictions & ~val_test_labels)
FN = np.sum(~predictions & val_test_labels)


# Calculate Accuracy, Sensitivity, and Specificity
accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f"Accuracy: {accuracy}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")