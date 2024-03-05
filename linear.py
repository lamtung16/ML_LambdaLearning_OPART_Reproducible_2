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
    def __init__(self, input_size=1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.initialize_parameters()

    def initialize_parameters(self):
        for param in self.parameters():
            init.constant_(param, 0)

    def forward(self, x):
        return self.linear(x)



# learn model
def linear_training(inputs_df, outputs_df, chosen_feature, n_ites=300):
    # inputs
    inputs = inputs_df.iloc[:, 1:][chosen_feature].to_numpy()
    inputs = torch.Tensor(inputs)

    # outputs:
    targets_low  = torch.Tensor(outputs_df['min.log.lambda'].to_numpy().reshape(-1,1))
    targets_high = torch.Tensor(outputs_df['max.log.lambda'].to_numpy().reshape(-1,1))
    outputs = torch.cat((targets_low, targets_high), dim=1)

    # prepare training dataset
    dataset    = TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Instantiate model, loss function and opimizer
    model = LinearModel()
    criterion = SquaredHingeLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for _ in range(n_ites):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model



# get lldas
def linear_evaluate(input_train_df, output_train_df, inputs_val_df, chosen_feature, n_ites):
    model = linear_training(input_train_df, output_train_df, chosen_feature, n_ites)
    inputs = inputs_val_df.iloc[:, 1:][chosen_feature].to_numpy()
    inputs = torch.Tensor(inputs)
    with torch.no_grad():
        lldas = model(inputs).numpy().reshape(-1)
    lldas_df = pd.DataFrame(list(zip(inputs_val_df['sequenceID'], lldas)), columns=['sequenceID', 'llda'])
    return lldas_df