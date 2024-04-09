import pandas as pd
import numpy as np
import random
from joblib import Parallel, delayed

from utility_functions import get_acc, add_row_to_csv
from MLP import mlp_evaluate



# dataset
# for dataset in ['detailed', 'systematic', 'cancer']:
for dataset in ['detailed', 'systematic']:

    # training data
    fold_path = 'training_data/' + dataset + '/folds.csv'
    inputs_path = 'training_data/' + dataset + '/inputs.csv'
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

    # number of folds
    n_folds = fold_df['fold'].nunique()

    # feature engineering transformation
    identity = lambda x: x
    log      = lambda x: np.log(x)
    loglog   = lambda x: np.log(np.log(x))




    def get_fold_dfs(n_fold, test_fold):
        # get indices
        train_folds = list(range(1, n_fold+1))
        train_folds.remove(test_fold)

        val_fold = random.choice(train_folds)
        
        subtrain_folds = train_folds
        subtrain_folds.remove(val_fold)

        # train
        train_inputs_df  = inputs_df[inputs_df['sequenceID'].isin(fold_df[fold_df['fold'] == train_folds]['sequenceID'])]
        train_outputs_df = outputs_df[outputs_df['sequenceID'].isin(fold_df[fold_df['fold'] == train_folds]['sequenceID'])]
        train_eval_df    = evaluation_df[evaluation_df['sequenceID'].isin(fold_df[fold_df['fold'] == train_folds]['sequenceID'])]


        # subtrain
        subtrain_inputs_df  = inputs_df[inputs_df['sequenceID'].isin(fold_df[fold_df['fold'] == subtrain_folds]['sequenceID'])]
        subtrain_outputs_df = outputs_df[outputs_df['sequenceID'].isin(fold_df[fold_df['fold'] == subtrain_folds]['sequenceID'])]
        subtrain_eval_df    = evaluation_df[evaluation_df['sequenceID'].isin(fold_df[fold_df['fold'] == subtrain_folds]['sequenceID'])]

        # validation
        val_inputs_df = inputs_df[inputs_df['sequenceID'].isin(fold_df[fold_df['fold'] == val_fold]['sequenceID'])]
        val_outputs_df = outputs_df[outputs_df['sequenceID'].isin(fold_df[fold_df['fold'] == val_fold]['sequenceID'])]
        val_eval_df = evaluation_df[evaluation_df['sequenceID'].isin(fold_df[fold_df['fold'] == val_fold]['sequenceID'])]

        # test
        test_inputs_df = inputs_df[inputs_df['sequenceID'].isin(fold_df[fold_df['fold'] == test_fold]['sequenceID'])]
        test_eval_df = evaluation_df[evaluation_df['sequenceID'].isin(fold_df[fold_df['fold'] == test_fold]['sequenceID'])]
        return train_inputs_df.sort_values(by='sequenceID'), train_outputs_df.sort_values(by='sequenceID'), subtrain_inputs_df.sort_values(by='sequenceID'), subtrain_outputs_df.sort_values(by='sequenceID'), subtrain_eval_df, val_inputs_df, val_eval_df, test_inputs_df, test_eval_df




    def process_combination(n_fold, test_fold, n_layer, layer_size, feature_dict, normalize):
        train_inputs_df, train_outputs_df, subtrain_inputs_df, subtrain_outputs_df, _, val_inputs_df, val_eval_df, test_inputs_df, test_eval_df = get_fold_dfs(n_fold, test_fold)
        chosen_feature = feature_dict['chosen_feature']
        f_engineer = feature_dict['f_engineer']
        is_f_engineer = 1 if f_engineer[0] != identity else 0
        lldas_eval_df = mlp_evaluate(
            input_train_df=subtrain_inputs_df,
            output_train_df=subtrain_outputs_df,
            inputs_val_df=val_inputs_df,
            hidden_layers=n_layer,
            hidden_size=layer_size,
            chosen_feature=chosen_feature,
            f_engineer=f_engineer,
            normalize=normalize
        )
        lldas_test_df = mlp_evaluate(
            input_train_df=train_inputs_df,
            output_train_df=train_outputs_df,
            inputs_val_df=test_inputs_df,
            hidden_layers=n_layer,
            hidden_size=layer_size,
            chosen_feature=chosen_feature,
            f_engineer=f_engineer,
            normalize=normalize
        )
        val_acc = get_acc(val_eval_df, lldas_eval_df)
        test_acc = get_acc(test_eval_df, lldas_test_df)
        return [test_fold, n_layer, layer_size, chosen_feature, is_f_engineer, normalize, val_acc, test_acc]




    n_layer_list = [1, 2, 3]
    layer_size_list = [2, 4, 8, 16, 32, 64, 128]
    feature_dict_list = [
        {'chosen_feature': ['length'], 'f_engineer': [loglog]},
        {'chosen_feature': ['length'], 'f_engineer': [identity]},
        {'chosen_feature': ['length', 'sd'], 'f_engineer': [loglog, log]},
        {'chosen_feature': ['length', 'sd'], 'f_engineer': [identity, identity]},
        {'chosen_feature': ['sd', 'range_value', 'length', 'sum_diff'], 'f_engineer': [log, log, loglog, log]},
        {'chosen_feature': ['sd', 'range_value', 'length', 'sum_diff'], 'f_engineer': [identity, identity, identity, identity]}
    ]




    # linear
    linear_results = Parallel(n_jobs=-1)(
        delayed(process_combination)(
            n_folds, test_fold, 0, 1, feature_dict, 0
        ) for test_fold in range(1, n_folds + 1)
        for feature_dict in feature_dict_list
    )

    for row in linear_results:
        add_row_to_csv(acc_rate_path, ['fold', 'n_layer', 'layer_size', 'features', 'f_engineer', 'normalize', 'val_acc', 'test_acc'], row)




    # non linear
    mlp_results = Parallel(n_jobs=-1)(
        delayed(process_combination)(
            n_folds, test_fold, n_layer, layer_size, feature_dict, 1
        ) for test_fold in range(1, n_folds + 1)
        for n_layer in n_layer_list
        for layer_size in layer_size_list
        for feature_dict in feature_dict_list
    )

    for row in mlp_results:
        add_row_to_csv(acc_rate_path, ['fold', 'n_layer', 'layer_size', 'features', 'f_engineer', 'normalize', 'val_acc', 'test_acc'], row)
