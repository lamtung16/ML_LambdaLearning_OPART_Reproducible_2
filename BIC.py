import numpy as np


def BIC(inputs_df):
    lldas_df = inputs_df[['sequenceID', 'n.loglog']]
    lldas_df = lldas_df.rename(columns={'n.loglog': 'llda'})
    return lldas_df