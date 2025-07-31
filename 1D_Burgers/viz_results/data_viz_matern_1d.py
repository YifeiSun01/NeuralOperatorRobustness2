import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import re

def plot_matern_diff_1d(index, data, cl, nu):
    for datum in data:
        if datum["index"] == index:
            all_data = np.concatenate([
                datum["a"].detach().cpu().numpy() if isinstance(datum["a"], torch.Tensor) else datum["a"],
                datum["g(a)"].detach().cpu().numpy() if isinstance(datum["g(a)"], torch.Tensor) else datum["g(a)"],
                datum["G(a)"].detach().cpu().numpy() if isinstance(datum["G(a)"], torch.Tensor) else datum["G(a)"],
            ])
            y_min = np.min(all_data)  # Global minimum value
            y_max = np.max(all_data)

            indices = np.arange(len(datum["a"]))/len(datum["a"])

            def calculate_metrics(true, pred):
                rmse = np.sqrt(np.mean((true - pred)**2))
                mae = np.mean(np.abs(true - pred))
                mape = np.mean(np.abs((true - pred) / (np.abs(true) + 1e-10))) * 100  # Avoid division by zero
                return rmse, mae, mape

            # Calculate all metrics
            metrics_orig = calculate_metrics(datum["g(a)"], datum["G(a)"])

            # Create figure
            fig, ((ax1)) = plt.subplots(1, 1, figsize=(16, 12))

            # Subplot 1: Unperturbed
            ax1.plot(indices, datum["a"], label='Input (original)')
            ax1.plot(indices, datum["g(a)"], label='PDE output (original)')
            ax1.plot(indices, datum["G(a)"], label='FNO output (original)')
            ax1.set_title(f'Different GRF Distribution Input FNO vs PDE: RMSE={metrics_orig[0]:.6f}, MAE={metrics_orig[1]:.6f}, MAPE={metrics_orig[2]:.1f}% of the two lines here')
            ax1.set_xlabel('Normalized x coordinate')
            ax1.set_ylabel('Values')
            ax1.legend()
            ax1.set_ylim(y_min, y_max)


            # Overall title and layout
            plt.tight_layout()
            plt.suptitle(f"1D Burgers Equation: FNO vs PDE Comparison\nG: FNO, g: PDE Different GRF Distribution Input", y=1.02, fontsize=14)
            
            # Save and show
            file_folder = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/perturbation_results/1D/plots/"
            os.makedirs(file_folder, exist_ok=True)
            filename = f"{file_folder}burgers_1d_fno_predictions_cl{cl}_nu{nu}_idx{index+1}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(filename)
            plt.close()
            # plt.show()

# Load the pickle file

if __name__ == "__main__":

    # print("Current directory:", os.getcwd())
    folder_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/perturbation_results/1D/pickle_files/"

    # List all files and directories
    extension = ".pkl"  # Example: Filter PNG files
    all_items = [f for f in os.listdir(folder_path) if f.endswith(extension) and "matern" in f]
    print(all_items)
    for item in all_items[:]:
        if "0.0005" in item:
            cl = re.findall(r'cl([a-zA-Z0-9|\.]+)', item)[0]
            nu = re.findall(r'nu([a-zA-Z0-9|\.]+)', item)[0]
            with open(f"{folder_path}{item}", "rb") as f:
                data = pickle.load(f)
            
            for index in [0,1,2,3,4]:
                plot_matern_diff_1d(index, data, cl, nu)
                # pass


                    
