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

    config.workdir = f"{base_path}perturbed_results_as_inputs/1D"

    config.device = select_optimal_gpu()


    return config