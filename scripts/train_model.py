# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %%
import os
import sys
import random

# %%
import pandas as pd
import numpy as np

# %%
from joblib import dump
from scipy.io import loadmat
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from metaod.models.gen_meta_features import generate_meta_features
from metaod.models.core import MetaODClass
from metaod.models.utility import fix_nan


# %%
@dataclass
class Model:
    name: str
    type: str
    pars: tuple

@dataclass
class Dataset:
    name: str
    path: str


# %%
def train_from_scratch(model_list, data_list):
    
    script_directory = os.path.dirname(__file__)
    excel_file_path = os.path.join(script_directory, '..', 'data', 'performance_table.xlsx')
    perf_df = pd.read_excel(excel_file_path, sheet_name='AP')    
    
    perf_mat = perf_df.to_numpy()
    perf_mat_red = fix_nan(perf_mat[:, 4:].astype('float'))
    
    n_datasets, n_configs = perf_mat.shape[0], perf_mat.shape[1]
    print('num_datasets:', n_datasets, '\nnum_configs:', n_configs)

    data_headers = perf_mat[:, 0]
    config_headers = perf_df.columns[4:]
    dump(config_headers, 'model_list.joblib')  
    
    meta_mat = np.zeros((n_datasets, 200))
    
    for index, dataset in enumerate(data_list):
        mat = loadmat('../data/' + dataset.path)
        X = mat['X']
        meta_mat[index, :], meta_vec_names = generate_meta_features(X)
        
    meta_scalar = MinMaxScaler()
    meta_mat_transformed = meta_scalar.fit_transform(meta_mat)
    meta_mat_transformed = fix_nan(meta_mat_transformed)
    from pathlib import Path
    #dump(meta_scalar, Path('results') / 'meta_scalar.joblib')
    
    
    # split data into train and valid
    seed = 0
    full_list = list(range(n_datasets))
    random.Random(seed).shuffle(full_list)
    n_train = int(0.85 * n_datasets)

    train_index = full_list[:n_train]
    valid_index = full_list[n_train:]

    train_set = perf_mat_red[train_index, :].astype('float64')
    valid_set = perf_mat_red[valid_index, :].astype('float64')

    train_meta = meta_mat_transformed[train_index, :].astype('float64')
    valid_meta = meta_mat_transformed[valid_index, :].astype('float64')
    
    train_meta[np.isnan(train_meta)] = 0
    valid_meta[np.isnan(valid_meta)] = 0
    
    max_n_components = min(meta_mat_transformed.shape[0], meta_mat_transformed.shape[1])

    # Choose a valid value for n_components based on your data
    #n_components = min(30, max_n_components)
    
    print(meta_mat_transformed.shape[0], meta_mat_transformed.shape[1])
    clf = MetaODClass(train_set, valid_performance=valid_set, n_factors=1,

                    learning='sgd')
    
    clf.train(n_iter=50, meta_features=train_meta, valid_meta=valid_meta,
            learning_rate=0.05, max_rate=0.9, min_rate=0.1, discount=1,
            n_steps=8)

    # output transformer (for meta-feature) and the trained clf
    dump(clf, Path('results')  /  str('train_' + str(seed) + '.joblib'))

# %%

# %%
if __name__ == '__main__':
    
    model_list = [Model('LODA (5, 10)', 'LODA', (5, 10)), Model('LOF (70, "euclidean")', 'LOF', (70, "euclidean"))]
    
    data_list = [Dataset('Annthyroid', 'annthyroid.mat'), Dataset('Arrhythmia', 'arrhythmia.mat')]
    
    train_from_scratch(model_list, data_list)
