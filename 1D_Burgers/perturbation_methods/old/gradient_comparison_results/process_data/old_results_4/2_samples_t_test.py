import pickle

# Open the pickle file in binary read mode
with open('gradient_comparison_norm2modified_exponax_nu0.0005.pkl', 'rb') as file:
    # Load the data from the pickle file
    data = pickle.load(file)

import scipy.stats as stats
from collections import defaultdict

def analyze_gradient_groups(data):
    """
    Groups data by all keys except index, then performs t-tests comparing final losses.
    
    Parameters:
    - data: The nested dictionary containing all experiment results
    
    Returns:
    - Dictionary containing grouped results and t-test outcomes
    """
    # Step 1: Group data by all keys except index
    groups = defaultdict(list)
    
    for key_tuple in data.keys():
        # Extract all parts of the key except index
        norm_type, index_part, numsteps, epsilon, alpha = key_tuple
        group_key = (norm_type, numsteps, epsilon, alpha)
        
        # Store the index and corresponding data
        groups[group_key].append((index_part, data[key_tuple]))
    
    # Step 2: Process each group and perform t-tests
    results = {}
    
    for group_key, group_data in groups.items():
        # Only process groups with at least 2 samples
        if len(group_data) < 2:
            continue
            
        # Extract final step losses for with_solver and without_solver
        with_solver_losses = []
        without_solver_losses = []
        
        for index_part, experiment_data in group_data:
            # Get the final step (step_99)
            final_step = experiment_data['step_update']['step_99']
            
            # Get the 'after' losses
            with_solver_loss = final_step['loss_metrics']['after']['with_solver']['combined_loss']
            without_solver_loss = final_step['loss_metrics']['after']['with_solver_on_without']['combined_loss']
            
            with_solver_losses.append(with_solver_loss)
            without_solver_losses.append(without_solver_loss)
        
        # Perform paired t-test (since these are paired samples - same input index)
        t_stat, p_value = stats.ttest_rel(with_solver_losses, without_solver_losses)
        
        # Calculate mean differences
        mean_diff = np.mean(np.array(with_solver_losses) - np.array(without_solver_losses))
        abs_mean_diff = np.mean(np.abs(np.array(with_solver_losses) - np.array(without_solver_losses)))
        
        # Store results
        results[group_key] = {
            'n_samples': len(group_data),
            'with_solver_mean': np.mean(with_solver_losses),
            'without_solver_mean': np.mean(without_solver_losses),
            'mean_difference': mean_diff,
            'abs_mean_difference': abs_mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'with_solver_losses': with_solver_losses,
            'without_solver_losses': without_solver_losses
        }
    
    return results

analyze_gradient_groups(data)