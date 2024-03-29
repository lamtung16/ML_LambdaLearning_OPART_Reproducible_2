import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utility_functions import SquaredHingeLoss, get_acc



# Define the linear model
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)



# Define the L1 regularization term
def l1_regularization(model, lambda_l1):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.norm(param, p=1)
    return lambda_l1 * l1_loss



# filter non-complete columns
def filter_numeric_columns(df):
    numeric_columns = ['sequenceID']
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            if df[column].notna().all():
                if not df[column].isin([np.inf, -np.inf]).any():
                    numeric_columns.append(column)
    numeric_columns = numeric_columns
    return df[numeric_columns]



# get fold dfs
def get_fold_dfs(fold, fold_df, inputs_df, outputs_df, evaluation_df):
    train_inputs_df = inputs_df[inputs_df['sequenceID'].isin(fold_df[fold_df['fold'] != fold]['sequenceID'])]
    train_outputs_df = outputs_df[outputs_df['sequenceID'].isin(fold_df[fold_df['fold'] != fold]['sequenceID'])]
    train_eval_df = evaluation_df[evaluation_df['sequenceID'].isin(fold_df[fold_df['fold'] != fold]['sequenceID'])]
    test_inputs_df = inputs_df[inputs_df['sequenceID'].isin(fold_df[fold_df['fold'] == fold]['sequenceID'])]
    test_eval_df = evaluation_df[evaluation_df['sequenceID'].isin(fold_df[fold_df['fold'] == fold]['sequenceID'])]
    return filter_numeric_columns(train_inputs_df), train_outputs_df, train_eval_df, filter_numeric_columns(test_inputs_df), test_eval_df



# main part
for dataset in ['cancer', 'detailed', 'systematic']:

    # training data
    fold_path = 'training_data/' + dataset + '/folds.csv'
    inputs_path = 'training_data/' + dataset + '/inputs_old.csv'
    outputs_path = 'training_data/' + dataset + '/outputs.csv'
    evaluation_path = 'training_data/' + dataset + '/evaluation.csv'

    # writing accuracy rate path
    acc_rate_path = 'acc_rate/' + dataset + '.csv'

    # path to write df to csv
    output_df_path = 'record_dataframe/' + dataset + '/'

    # raw dfs
    fold_df = pd.read_csv(fold_path)
    inputs_df = pd.read_csv(inputs_path)
    outputs_df = pd.read_csv(outputs_path)
    evaluation_df = pd.read_csv(evaluation_path)

    total_acc = 0
    for fold in range(1, 7):
        train_inputs_df, train_outputs_df, train_eval_df, test_inputs_df, test_eval_df = get_fold_dfs(fold, fold_df, inputs_df, outputs_df, evaluation_df)

        # inputs
        inputs = train_inputs_df.drop(columns=['sequenceID']).to_numpy()
        inputs = torch.Tensor(inputs)

        test_inputs = test_inputs_df.drop(columns=['sequenceID']).to_numpy()
        test_inputs = torch.Tensor(test_inputs)

        # outputs
        targets_low  = torch.Tensor(train_outputs_df['min.log.lambda'].to_numpy().reshape(-1,1))
        targets_high = torch.Tensor(train_outputs_df['max.log.lambda'].to_numpy().reshape(-1,1))
        outputs = torch.cat((targets_low, targets_high), dim=1)

        # Hyper
        lambda_l1 = 0.01
        lr = 0.00001
        n_iters = 10000000

        # Initialize the model
        model = LinearModel(inputs.shape[1])

        # Define loss function and optimizer
        criterion = SquaredHingeLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        best_loss = float('inf')
        patience = 100000
        wait = 0
        for epoch in range(n_iters):
            # Forward pass
            model.train()
            loss = criterion(model(inputs), outputs)
            
            # Add L1 regularization to the loss
            loss += l1_regularization(model, lambda_l1)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check if the loss has decreased
            if loss < best_loss:
                best_loss = loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping after {} epochs without improvement.".format(patience))
                    break

        # get acc
        with torch.no_grad():
            lldas = model(test_inputs).numpy().reshape(-1)

        lldas_df = pd.DataFrame(list(zip(test_inputs_df['sequenceID'], lldas)), columns=['sequenceID', 'llda'])
        acc = get_acc(test_eval_df, lldas_df)
        total_acc += acc
        print(dataset, fold, acc)
    
    # print avg
    print(dataset, total_acc/6, "\n")




# detailed 1 95.983606557377051
# detailed 2 94.821974965229485
# detailed 3 95.77304964539007
# detailed 4 95.91816920943134
# detailed 5 96.396481732070365
# detailed 6 96.535666218034994
# detailed 95.76945168034249 

# systematic 1 93.28070175438596
# systematic 2 92.192982456140353
# systematic 3 94.912280701754385
# systematic 4 93.6842105263158
# systematic 5 93.72231985940246
# systematic 6 93.16871704745167
# systematic 93.96945168034249