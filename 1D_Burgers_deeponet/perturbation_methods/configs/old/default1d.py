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

    config.attack_method = "PGD without solver gradient"

    config.device = select_optimal_gpu()

    config.num_samples = 10000

    config.burgers = burgers = ml_collections.ConfigDict()
    burgers.epochs = 0.0005
    # config.num_steps_list = [10]
    # config.num_steps_list = [2,5,10,20]
    # config.num_steps_list = [2,5,10,20,50]
    config.num_steps_list = [2,5,10,20,50,100]
    # config.num_steps_list = [2,5,10,20,50,100,200,500]
    # config.epsilon_list = [0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07]
    # config.epsilon_list = [0.1, 0.2, 0.5, 0.7, 1]
    # config.epsilon_list = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1]
    # config.epsilon_list = [0.01, 0.03, 0.05, 0.07, 0.3, 0.7]
    # config.epsilon_list = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007]
    config.epsilon_list = [1.3, 1.5, 1.7, 2, 2.3, 2.5, 2.7, 3]

    return config