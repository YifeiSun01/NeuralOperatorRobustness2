import ml_collections
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from perturbation_methods.configs.gpu_choose import select_optimal_gpu

def get_default_configs():

    config = ml_collections.ConfigDict()

    base_path = "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/"

    config.dict_input_path = f"{base_path}perturbation_methods/x_y_dict/1D/Burgers/inputs_dim1d_nx1024_N20000_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.0005_t1.0_seed45.pt"

    config.dict_output_path = f"{base_path}perturbation_methods/x_y_dict/1D/Burgers/outputs_dim1d_nx1024_N20000_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.0005_t1.0_seed45.pt"

    config.model_path = f"{base_path}saved_models/1D/modes16_width64_epochs500/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.0005_t1.0_seed45.pth"

    config.workdir = f"{base_path}perturbation_results/1D"

    config.device = select_optimal_gpu()

    config.num_records = 10

    config.burgers = burgers = ml_collections.ConfigDict()
    burgers.epochs = 0.0005
    
    """
    config.inputs = [   
        ("inf",0.1,100,0.001), 
        ("inf",0.1,100,0.005),
        ("inf",0.1,100,0.01), 
        ("inf",0.1,100,0.05), 
        ("inf",0.1,100,0.1), 
        ("inf",0.5,100,0.005), 
        ("inf",0.5,100,0.025), 
        ("inf",0.5,100,0.05), 
        ("inf",0.5,100,0.25), 
        ("inf",0.5,100,0.5), 

        (2,10,100,0.01), 
        (2,10,100,0.05), 
        (2,10,100,0.1), 
        (2,10,100,0.5), 
        (2,20,100,0.02), 
        (2,20,100,0.1), 
        (2,20,100,0.2), 
        (2,20,100,1), 
        ]
    """
    
    config.inputs = [   
        ("inf",0.1,100,0.005),
        ("inf",0.5,100,0.025),
        ("inf",0.1,100,0.002),
        ("inf",0.5,100,0.01),
        (2,10,100,0.5), 
        (2,20,100,1), 
        (2,10,100,0.2), 
        (2,20,100,0.5), 
        ]


    return config