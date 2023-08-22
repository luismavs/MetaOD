import os
import sys
import random

import pandas as pd
import numpy as np

from joblib import dump
from scipy.io import loadmat
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from metaod.models.gen_meta_features import generate_meta_features
from metaod.models.core import MetaODClass
from metaod.models.utility import fix_nan

@dataclass
class Model:
    name: str
    type: str
    pars: tuple

@dataclass
class Dataset:
    name: str
    path: str


def train_from_scratch(model_list, data_list):
 
if __name__ == '__main__':
    
    model_list = [Model('LODA (5, 10)', 'LODA', (5, 10)), Model('LOF (70, "euclidean")', 'LOF', (70, "euclidean"))]
    
    data_list = [Dataset('Annthyroid', 'annthyroid.mat'), Dataset('Arrhythmia', 'arrhythmia.mat')]
    
    train_from_scratch(model_list, data_list)
