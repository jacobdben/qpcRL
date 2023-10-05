# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:49:23 2019

@author: Triton4acq
"""

from qcodes.dataset.data_set import load_by_id, DataSet
import pandas as pd
from pprint import pprint  # for pretty-printing python variables like 'dict'
import numpy as np
import os

def export_by_id(run_id,npath):
    dataset = load_by_id(run_id)
    columns = []
    for param in dataset.get_parameters():
        values = np.ravel(dataset.get_data(param.name))
        columns.append(values)
    data = np.array(columns).T
    np.savetxt(npath, data)

        

def export_by_id_pd(run_id, npath):
   dataset = load_by_id(run_id)
   dfdict = dataset.to_pandas_dataframe_dict()
   dfs_to_save = list()
   for parametername, df in dfdict.items():
       dfs_to_save.append(df)

   df_to_save = pd.concat(dfs_to_save, axis=1)
   df_to_save.to_csv(path_or_buf=npath, header=False, sep='\t')
   
   
def export_snapshot_by_id(run_id,npath):
    dataset = load_by_id(run_id)
    snap=dataset.snapshot
    with open(npath, 'wt') as f:
        pprint(snap, stream=f)
        
