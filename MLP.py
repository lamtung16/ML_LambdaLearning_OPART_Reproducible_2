import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from utility_functions import SquaredHingeLoss
from torch.utils.data import DataLoader, TensorDataset




# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size):
        super(MLPModel, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)
        
        self.initialize_parameters()

    def initialize_parameters(self):
        for param in self.parameters():
            init.constant_(param, 0.5)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x



# normalzie
def normalize_data(tensor):
    # Calculate mean and standard deviation along the feature dimension
    mean = torch.mean(tensor, dim=0)
    std = torch.std(tensor, dim=0)

    # Normalize the tensor
    normalized_tensor = (tensor - mean) / std

    return normalized_tensor



# learn model
def mlp_training(inputs_df, outputs_df, hidden_layers, hidden_size, chosen_feature, f_engineer, n_ites=500):
    # inputs
    inputs = inputs_df[chosen_feature].to_numpy()

    # feature engineering
    for i in range(len(f_engineer)):
        inputs[:, i] = f_engineer[i](inputs[:, i])
    inputs = torch.Tensor(inputs)

    # normalize input
    inputs = normalize_data(inputs)

    # outputs
    targets_low  = torch.Tensor(outputs_df['min.log.lambda'].to_numpy().reshape(-1,1))
    targets_high = torch.Tensor(outputs_df['max.log.lambda'].to_numpy().reshape(-1,1))
    outputs = torch.cat((targets_low, targets_high), dim=1)

    # prepare training dataset
    dataset    = TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Instantiate model, loss function and opimizer
    model = MLPModel(inputs.shape[1], hidden_layers, hidden_size)
    criterion = SquaredHingeLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for _ in range(n_ites):
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model



def cv_learn(n_splits, X, y, n_hiddens, layer_size, batch_size, n_ites):
    
    # Define the K-fold cross-validation
    kf = KFold(n_splits)

    # loss function
    loss_func = SquaredHingeLoss()

    # learn best ite
    total_losses = {'subtrain': np.zeros(n_ites), 'val': np.zeros(n_ites)}
    data_splits = {'X_subtrain': [], 'X_val': [], 'y_subtrain': [], 'y_val': []}
    
    for subtrain_idx, val_idx in kf.split(X):

        # Split the data into training and validation sets
        indices = {'subtrain': subtrain_idx, 'val': val_idx}
        for key in data_splits.keys():
            feature_target, set_type = key.split('_')       # (X or y) and (subtrain or val)
            data_splits[key].append(X[indices[set_type]] if feature_target == 'X' else y[indices[set_type]])

        # Create DataLoader
        dataset    = TensorDataset(data_splits['X_subtrain'][-1], data_splits['y_subtrain'][-1])
        dataloader = DataLoader(dataset, batch_size, shuffle=False)

        # Define your model
        model = MLPModel(X.shape[1], n_hiddens, layer_size)

        # define optimizer
        optimizer = optim.Adam(model.parameters())

        # Training loop for the specified number of iterations
        for i in range(n_ites):
            # training
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                loss = loss_func(model(inputs), labels)
                loss.backward()
                optimizer.step()

            # validating
            model.eval()
            with torch.no_grad():
                val_loss = loss_func(model(data_splits['X_val'][-1]), data_splits['y_val'][-1])

            # add train_loss and val_loss into arrays
            total_losses['val'][i] += val_loss.item()

    best_no_ite = np.argmin(total_losses['val'])
    return best_no_ite + 1



# get lldas
def mlp_evaluate(input_train_df, output_train_df, inputs_val_df, hidden_layers, hidden_size, chosen_feature, f_engineer, n_ites):
    # cross-validation to get best number of iterations
    
    # cv inputs
    cv_inputs = input_train_df[chosen_feature].to_numpy()
    # feature engineering
    for i in range(len(f_engineer)):
        cv_inputs[:, i] = f_engineer[i](cv_inputs[:, i])
    cv_inputs = torch.Tensor(cv_inputs)
    
    # cv output
    targets_low  = torch.Tensor(output_train_df['min.log.lambda'].to_numpy().reshape(-1,1))
    targets_high = torch.Tensor(output_train_df['max.log.lambda'].to_numpy().reshape(-1,1))
    cv_outputs = torch.cat((targets_low, targets_high), dim=1)
    best_n_ite = cv_learn(2, cv_inputs, cv_outputs, hidden_layers, hidden_size, 1, n_ites)
    
    # trained model
    model = mlp_training(input_train_df, output_train_df, hidden_layers, hidden_size, chosen_feature, f_engineer, best_n_ite)
    
    # test inputs
    inputs = inputs_val_df[chosen_feature].to_numpy()

    # feature engineering
    for i in range(len(f_engineer)):
        inputs[:, i] = f_engineer[i](inputs[:, i])
    inputs = torch.Tensor(inputs)
    
    # normalize inputs
    inputs = normalize_data(inputs)
    
    with torch.no_grad():
        lldas = model(inputs).numpy().reshape(-1)
    lldas_df = pd.DataFrame(list(zip(inputs_val_df['sequenceID'], lldas)), columns=['sequenceID', 'llda'])
    return lldas_df