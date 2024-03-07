import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import pandas as pd
from utility_functions import SquaredHingeLoss
from torch.utils.data import DataLoader, TensorDataset



# Define the linear model
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.initialize_parameters()

    def initialize_parameters(self):
        for param in self.parameters():
            init.constant_(param, 0)

    def forward(self, x):
        return self.linear(x)



# learn model
def linear_training(inputs_df, outputs_df, chosen_feature, f_engineer, batch_size, margin, n_ites, verbose):
    # inputs
    inputs = inputs_df[chosen_feature].to_numpy()
    for i in range(len(f_engineer)):
        inputs[:, i] = f_engineer[i](inputs[:, i])
    inputs = torch.Tensor(inputs)

    # outputs:
    targets_low  = torch.Tensor(outputs_df['min.log.lambda'].to_numpy().reshape(-1,1))
    targets_high = torch.Tensor(outputs_df['max.log.lambda'].to_numpy().reshape(-1,1))
    outputs = torch.cat((targets_low, targets_high), dim=1)

    # prepare training dataset
    dataset    = TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # Instantiate model, loss function and optimizer
    model = LinearModel(inputs.shape[1])
    criterion = SquaredHingeLoss(margin)
    optimizer = optim.Adam(model.parameters())

    # Initialize early stopping parameters
    best_loss = float('inf')
    patience = 5  # Number of epochs to wait before early stopping
    num_bad_epochs = 0

    # Training loop
    for i in range(n_ites):
        for batch_input, batch_output in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_input), batch_output)
            loss.backward()
            optimizer.step()

        # Calculate validation loss
        val_loss = criterion(model(inputs), outputs)

        if verbose==1:
            print(f"{i}, loss: {val_loss}")

        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
            if num_bad_epochs >= patience:
                if verbose==1:
                    print(f"Stopping early at epoch {i}, loss: {val_loss}")
                break

    return model



# get lldas
def linear_evaluate(input_train_df, output_train_df, inputs_val_df, chosen_feature, f_engineer, n_ites):
    model = linear_training(input_train_df, output_train_df, chosen_feature, f_engineer, n_ites)
    inputs = inputs_val_df[chosen_feature].to_numpy()
    for i in range(len(f_engineer)):
        inputs[:, i] = f_engineer[i](inputs[:, i])
    inputs = torch.Tensor(inputs)
    with torch.no_grad():
        lldas = model(inputs).numpy().reshape(-1)
    lldas_df = pd.DataFrame(list(zip(inputs_val_df['sequenceID'], lldas)), columns=['sequenceID', 'llda'])
    return lldas_df