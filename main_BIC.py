import pandas as pd
from utility_functions import get_acc
from BIC import BIC




# for dataset in ['cancer', 'detailed', 'systematic']:
for dataset in ['detailed', 'systematic']:

    # training data
    fold_path = 'training_data/' + dataset + '/folds.csv'
    inputs_path = 'training_data/' + dataset + '/inputs.csv'
    evaluation_path = 'training_data/' + dataset + '/evaluation.csv'

    # raw dfs
    fold_df = pd.read_csv(fold_path)
    inputs_df = pd.read_csv(inputs_path)
    evaluation_df = pd.read_csv(evaluation_path)

    # number of folds
    n_folds = fold_df['fold'].nunique()

    # main function
    total_acc = 0
    for fold in range(1, n_folds + 1):
        fold_inputs_df = inputs_df[inputs_df['sequenceID'].isin(fold_df[fold_df['fold'] == fold]['sequenceID'])]
        fold_eval_df = evaluation_df[evaluation_df['sequenceID'].isin(fold_df[fold_df['fold'] == fold]['sequenceID'])]

        lldas_df = BIC(fold_inputs_df)
        total_acc += get_acc(fold_eval_df, lldas_df)
    print("dataset: %8s \t acc: %f" % (dataset, total_acc/6))


# dataset: detailed        acc: 86.023357
# dataset: systematic      acc: 91.983268