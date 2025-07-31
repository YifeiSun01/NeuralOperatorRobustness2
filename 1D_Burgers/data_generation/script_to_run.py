from Burgers1d_generation import generate_burgers_dataset

# for nu in [0.005, 0.0005]:
# for nu in [0.1, 0.01, 0.001]:
# for nu in [0.01, 0.001]:
for nu in [0.05,0.005]:
    grf_params = {
        'dim': 1,
        'nx': 1024,
        'kernel': 'gaussian',
        'kernel_params': {'correlation_length': 0.03},
        'bc': 'periodic',
        'seed': 45,
        'zero_mean':False
    }

    burger_params = {
        'nu': nu,
        'simulation_time': 1.0,
        "step": 0.001
    }

    dataset_params = {
        "num_record": 1500
    }

    params = {
        "GRF": grf_params,
        "dataset": dataset_params,
        "burgers": burger_params,
    }

    # for solver_name in ["exponax", "phiflow"]: 
    for solver_name in ["exponax"]:
        generate_burgers_dataset(params, save_dir="datasets/1D/Burgers", solver_name=solver_name)