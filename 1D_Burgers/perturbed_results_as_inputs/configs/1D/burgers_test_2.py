import torch
import ml_collections
# from configs.default1d_cmwno import get_default_configs
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from perturbation_methods.configs.default1d import get_default_configs

def get_config():

    config = config = get_default_configs()


    return config