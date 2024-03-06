import numpy as np
import pandas as pd


def BIC(inputs_df):
    seqID = inputs_df['sequenceID']
    lldas = np.log(np.log(inputs_df['length']))
    lldas_df = pd.concat([seqID, lldas], axis=1)
    return lldas_df.rename(columns={'length': 'llda'})