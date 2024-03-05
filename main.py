import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from utility_functions import get_acc, record
from BIC import BIC
from linear import linear_evaluate


def get_stat(dataset):

    # training data
    inputs_path = 'training_data/' + dataset + '/inputs.csv'
    outputs_path = 'training_data/' + dataset + '/outputs.csv'
    evaluation_path = 'training_data/' + dataset + '/evaluation.csv'

    # writing accuracy rate path
    acc_rate_path = 'acc_rate/' + dataset + '.csv'

    # path to write df to csv
    output_df_path = 'record_dataframe/' + dataset + '/'

    # raw dfs
    evaluation_df = pd.read_csv(evaluation_path)
    inputs_df = pd.read_csv(inputs_path)
    outputs_df = pd.read_csv(outputs_path)


    # fold dfs
    fold_size = int(inputs_df.shape[0]/2)

    inputs_fold1_df = inputs_df[0:fold_size]
    inputs_fold2_df = inputs_df[fold_size:]

    outputs_fold1_df = outputs_df[0:fold_size]
    outputs_fold2_df = outputs_df[fold_size:]

    evaluation_fold1_df = evaluation_df[evaluation_df['sequenceID'].isin(inputs_fold1_df['sequenceID'])]
    evaluation_fold2_df = evaluation_df[evaluation_df['sequenceID'].isin(inputs_fold2_df['sequenceID'])]

    # methods = ['BIC.1', 'linear.2', 'linear.6', 'mlp.1.8', 'mlp.2.16']
    methods = ['BIC.1', 'linear.2']
    for method in methods:
        if(method == 'BIC.1'):
            lldas_test_fold1_df = delayed(BIC)(inputs_fold1_df)
            lldas_test_fold2_df = delayed(BIC)(inputs_fold2_df)
            
        elif(method == 'linear.2'):
            chosen_feature = ['n.loglog']
            lldas_test_fold1_df = delayed(linear_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, chosen_feature, 100)
            lldas_test_fold2_df = delayed(linear_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, chosen_feature, 100)

        lldas_test_fold1_df, lldas_test_fold2_df = Parallel(n_jobs=2)([lldas_test_fold1_df, lldas_test_fold2_df])
        acc_test_fold1 = get_acc(evaluation_fold1_df, lldas_test_fold1_df)
        acc_test_fold2 = get_acc(evaluation_fold2_df, lldas_test_fold2_df)
        record(method, lldas_test_fold1_df, lldas_test_fold2_df, acc_test_fold1, acc_test_fold2, output_df_path, acc_rate_path)


stat_dataset_1 = delayed(get_stat)('detailed')
stat_dataset_2 = delayed(get_stat)('systematic')
stat_dataset_1, stat_dataset_2 = Parallel(n_jobs=2)([stat_dataset_1, stat_dataset_2])