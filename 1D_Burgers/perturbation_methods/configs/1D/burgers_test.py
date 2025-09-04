import torch
import ml_collections
# from configs.default1d_cmwno import get_default_configs
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from perturbation_methods.configs.default1d import get_default_configs

def get_config():

    config = config = get_default_configs()
    """
    config.inputs = [   
        ("inf",0.1,10,0.005),
        ("inf",0.5,10,0.025),
        ("inf",0.1,10,0.002),
        ("inf",0.5,10,0.01),
        ("inf",0.1,20,0.005),
        ("inf",0.5,20,0.025),
        ("inf",0.1,20,0.002),
        ("inf",0.5,20,0.01),
        (2,10,10,0.5), 
        (2,20,10,1), 
        (2,10,10,0.2), 
        (2,20,10,0.5), 
        (2,10,20,0.5), 
        (2,20,20,1), 
        (2,10,20,0.2), 
        (2,20,20,0.5), 
        ]
    """


    return config