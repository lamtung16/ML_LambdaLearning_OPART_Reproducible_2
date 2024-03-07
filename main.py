import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from utility_functions import get_acc, record
from BIC import BIC
from linear import linear_evaluate
from MLP import mlp_evaluate


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


    # feature engineering transformation
    identity = lambda x: x
    log      = lambda x: np.log(x)
    loglog   = lambda x: np.log(np.log(x))

    # job list
    job_list = [
        delayed(BIC)(inputs_fold1_df),
        delayed(BIC)(inputs_fold2_df),

        delayed(linear_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, ['length'], [loglog], 200),
        delayed(linear_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, ['length'], [loglog], 200),

        delayed(linear_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, ['length'], [identity], 200),
        delayed(linear_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, ['length'], [identity], 200),

        delayed(linear_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, ['sd', 'range_value', 'length', 'sum_diff'], [log, log, loglog, log], 200),
        delayed(linear_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, ['sd', 'range_value', 'length', 'sum_diff'], [log, log, loglog, log], 200),

        delayed(linear_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, ['sd', 'range_value', 'length', 'sum_diff'], [identity, identity, identity, identity], 200),
        delayed(linear_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, ['sd', 'range_value', 'length', 'sum_diff'], [identity, identity, identity, identity], 200),

        delayed(mlp_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, 1, 8,  ['sd', 'range_value', 'length', 'sum_diff'], [log, log, loglog, log], 500),
        delayed(mlp_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, 1, 8,  ['sd', 'range_value', 'length', 'sum_diff'], [log, log, loglog, log], 500),

        delayed(mlp_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, 1, 8,  ['sd', 'range_value', 'length', 'sum_diff'], [identity, identity, identity, identity], 500),
        delayed(mlp_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, 1, 8,  ['sd', 'range_value', 'length', 'sum_diff'], [identity, identity, identity, identity], 500),

        delayed(mlp_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, 2, 16, ['sd', 'range_value', 'length', 'sum_diff'], [log, log, loglog, log], 500),
        delayed(mlp_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, 2, 16, ['sd', 'range_value', 'length', 'sum_diff'], [log, log, loglog, log], 500),

        delayed(mlp_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, 2, 16, ['sd', 'range_value', 'length', 'sum_diff'], [identity, identity, identity, identity], 500),
        delayed(mlp_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, 2, 16, ['sd', 'range_value', 'length', 'sum_diff'], [identity, identity, identity, identity], 500),

        delayed(mlp_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, 3, 32, ['sd', 'range_value', 'length', 'sum_diff'], [log, log, loglog, log], 500),
        delayed(mlp_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, 3, 32, ['sd', 'range_value', 'length', 'sum_diff'], [log, log, loglog, log], 500),

        delayed(mlp_evaluate)(inputs_fold2_df, outputs_fold2_df, inputs_fold1_df, 3, 32, ['sd', 'range_value', 'length', 'sum_diff'], [identity, identity, identity, identity], 500),
        delayed(mlp_evaluate)(inputs_fold1_df, outputs_fold1_df, inputs_fold2_df, 3, 32, ['sd', 'range_value', 'length', 'sum_diff'], [identity, identity, identity, identity], 500),
    ]
    lldas_test_fold1_df_BIC,          lldas_test_fold2_df_BIC,         \
    lldas_test_fold1_df_linear_2,     lldas_test_fold2_df_linear_2,    \
    lldas_test_fold1_df_linear_nfe_2, lldas_test_fold2_df_linear_nfe_2,\
    lldas_test_fold1_df_linear_5,     lldas_test_fold2_df_linear_5,    \
    lldas_test_fold1_df_linear_nfe_5, lldas_test_fold2_df_linear_nfe_5,\
    lldas_test_fold1_df_mlp_1_8,      lldas_test_fold2_df_mlp_1_8,     \
    lldas_test_fold1_df_mlp_nfe_1_8,  lldas_test_fold2_df_mlp_nfe_1_8, \
    lldas_test_fold1_df_mlp_2_16,     lldas_test_fold2_df_mlp_2_16,    \
    lldas_test_fold1_df_mlp_nfe_2_16, lldas_test_fold2_df_mlp_nfe_2_16,\
    lldas_test_fold1_df_mlp_3_32,     lldas_test_fold2_df_mlp_3_32,    \
    lldas_test_fold1_df_mlp_nfe_3_32, lldas_test_fold2_df_mlp_nfe_3_32 \
    = Parallel(n_jobs=len(job_list))(job_list)

    record('BIC.1',        lldas_test_fold1_df_BIC,          lldas_test_fold2_df_BIC,          get_acc(evaluation_fold1_df, lldas_test_fold1_df_BIC),          get_acc(evaluation_fold2_df, lldas_test_fold2_df_BIC),          output_df_path, acc_rate_path)
    record('linear.2',     lldas_test_fold1_df_linear_2,     lldas_test_fold2_df_linear_2,     get_acc(evaluation_fold1_df, lldas_test_fold1_df_linear_2),     get_acc(evaluation_fold2_df, lldas_test_fold2_df_linear_2),     output_df_path, acc_rate_path)
    record('linear.2.nfe', lldas_test_fold1_df_linear_nfe_2, lldas_test_fold2_df_linear_nfe_2, get_acc(evaluation_fold1_df, lldas_test_fold1_df_linear_nfe_2), get_acc(evaluation_fold2_df, lldas_test_fold2_df_linear_nfe_2), output_df_path, acc_rate_path)
    record('linear.5',     lldas_test_fold1_df_linear_5,     lldas_test_fold2_df_linear_5,     get_acc(evaluation_fold1_df, lldas_test_fold1_df_linear_5),     get_acc(evaluation_fold2_df, lldas_test_fold2_df_linear_5),     output_df_path, acc_rate_path)
    record('linear.5.nfe', lldas_test_fold1_df_linear_nfe_5, lldas_test_fold2_df_linear_nfe_5, get_acc(evaluation_fold1_df, lldas_test_fold1_df_linear_nfe_5), get_acc(evaluation_fold2_df, lldas_test_fold2_df_linear_nfe_5), output_df_path, acc_rate_path)
    record('mlp.1.8',      lldas_test_fold1_df_mlp_1_8,      lldas_test_fold2_df_mlp_1_8,      get_acc(evaluation_fold1_df, lldas_test_fold1_df_mlp_1_8),      get_acc(evaluation_fold2_df, lldas_test_fold2_df_mlp_1_8),      output_df_path, acc_rate_path)
    record('mlp.1.8.nfe',  lldas_test_fold1_df_mlp_nfe_1_8,  lldas_test_fold2_df_mlp_nfe_1_8,  get_acc(evaluation_fold1_df, lldas_test_fold1_df_mlp_nfe_1_8),  get_acc(evaluation_fold2_df, lldas_test_fold2_df_mlp_nfe_1_8),  output_df_path, acc_rate_path)
    record('mlp.2.16',     lldas_test_fold1_df_mlp_2_16,     lldas_test_fold2_df_mlp_2_16,     get_acc(evaluation_fold1_df, lldas_test_fold1_df_mlp_2_16),     get_acc(evaluation_fold2_df, lldas_test_fold2_df_mlp_2_16),     output_df_path, acc_rate_path)
    record('mlp.2.16.nfe', lldas_test_fold1_df_mlp_nfe_2_16, lldas_test_fold2_df_mlp_nfe_2_16, get_acc(evaluation_fold1_df, lldas_test_fold1_df_mlp_nfe_2_16), get_acc(evaluation_fold2_df, lldas_test_fold2_df_mlp_nfe_2_16), output_df_path, acc_rate_path)
    record('mlp.3.32',     lldas_test_fold1_df_mlp_3_32,     lldas_test_fold2_df_mlp_3_32,     get_acc(evaluation_fold1_df, lldas_test_fold1_df_mlp_3_32),     get_acc(evaluation_fold2_df, lldas_test_fold2_df_mlp_3_32),     output_df_path, acc_rate_path)
    record('mlp.3.32.nfe', lldas_test_fold1_df_mlp_nfe_3_32, lldas_test_fold2_df_mlp_nfe_3_32, get_acc(evaluation_fold1_df, lldas_test_fold1_df_mlp_nfe_3_32), get_acc(evaluation_fold2_df, lldas_test_fold2_df_mlp_nfe_3_32), output_df_path, acc_rate_path)


stat_dataset_1 = delayed(get_stat)('detailed')
stat_dataset_2 = delayed(get_stat)('systematic')
stat_dataset_1, stat_dataset_2 = Parallel(n_jobs=2)([stat_dataset_1, stat_dataset_2])