
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import os

def plot_perturb_diff_1d(epsilon, index, data, perturb):
    for datum in data:
        if datum["epsilon"] == epsilon and datum["index"] == index:
            # print(epsilon, index)
        # if datum["epsilon"] == epsilon:
            # print({"index":datum["index"],"epsilon":datum["epsilon"],"num_steps":datum["num_steps"],"alpha":datum["alpha"],})
            all_data = np.concatenate([
                datum["a"].detach().cpu().numpy() if isinstance(datum["a"], torch.Tensor) else datum["a"],
                datum["g(a)"].detach().cpu().numpy() if isinstance(datum["g(a)"], torch.Tensor) else datum["g(a)"],
                datum["G(a)"].detach().cpu().numpy() if isinstance(datum["G(a)"], torch.Tensor) else datum["G(a)"],
                datum["delta"].detach().cpu().numpy() if isinstance(datum["delta"], torch.Tensor) else datum["delta"],
                np.squeeze(datum["a+delta"].detach().cpu().numpy()) if isinstance(datum["a+delta"], torch.Tensor) else np.squeeze(datum["a+delta"]),
                datum["g(a+delta)"].detach().cpu().numpy() if isinstance(datum["g(a+delta)"], torch.Tensor) else datum["g(a+delta)"],
                datum["G(a+delta)"].detach().cpu().numpy() if isinstance(datum["G(a+delta)"], torch.Tensor) else datum["G(a+delta)"]
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
            metrics_pert = calculate_metrics(datum["g(a+delta)"], datum["G(a+delta)"])
            metrics_resp = calculate_metrics(datum["g(a+delta)"]-datum["g(a)"], 
                                        datum["G(a+delta)"]-datum["G(a)"])
            metrics_diff = calculate_metrics(datum["G(a)"]-datum["g(a)"], 
                                        datum["G(a+delta)"]-datum["g(a+delta)"])

            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Subplot 1: Unperturbed
            ax1.plot(indices, datum["a"], label='Input (original)')
            ax1.plot(indices, datum["g(a)"], label='PDE output (original)')
            ax1.plot(indices, datum["G(a)"], label='FNO output (original)')
            ax1.set_title(f'Unperturbed\nFNO vs PDE: RMSE={metrics_orig[0]:.6f}, MAE={metrics_orig[1]:.6f}, MAPE={metrics_orig[2]:.1f}% of the two lines here')
            ax1.set_xlabel('Normalized x coordinate')
            ax1.set_ylabel('Values')
            ax1.legend()
            ax1.set_ylim(y_min, y_max)

            # Subplot 2: Perturbed
            ax2.plot(indices, np.squeeze(datum["a+delta"]), label='Input (perturbed)')
            ax2.plot(indices, datum["g(a+delta)"], label='PDE output (perturbed)')
            ax2.plot(indices, datum["G(a+delta)"], label='FNO output (perturbed)')
            ax2.set_title(f'Perturbed\nFNO vs PDE: RMSE={metrics_pert[0]:.6f}, MAE={metrics_pert[1]:.6f}, MAPE={metrics_pert[2]:.1f}% of the two lines here')
            ax2.set_xlabel('Normalized x coordinate')
            ax2.set_ylabel('Values')
            ax2.legend()
            ax2.set_ylim(y_min, y_max)

            # Subplot 3: Perturbation Responses
            ax3.plot(indices, datum["delta"], label='Input pertubation (delta)')
            ax3.plot(indices, datum["g(a+delta)"]-datum["g(a)"], label='PDE output pertubation (g(a+delta)-g(a))')
            ax3.plot(indices, datum["G(a+delta)"]-datum["G(a)"], label='FNO output pertubation (G(a+delta)-G(a))')
            ax3.set_title(f'Perturbation Responses\nRMSE={metrics_resp[0]:.6f}, MAE={metrics_resp[1]:.6f}, MAPE={metrics_resp[2]:.1f}% of the two lines here')
            ax3.set_xlabel('Normalized x coordinate')
            ax3.set_ylabel('Change Values')
            ax3.legend()

            # Subplot 4: FNO-PDE Differences
            ax4.plot(indices, datum["delta"], label='Input pertubation (delta)')
            ax4.plot(indices, datum["G(a)"]-datum["g(a)"], label='FNO-PDE output difference original (G(a)-g(a))')
            ax4.plot(indices, datum["G(a+delta)"]-datum["g(a+delta)"], label='FNO-PDE output difference perturbed G(a+delta)-g(a+delta)')
            ax4.set_title(f'FNO-PDE Differences \nRMSE={metrics_diff[0]:.6f}, MAE={metrics_diff[1]:.6f}, MAPE={metrics_diff[2]:.1f}% of the two lines here')
            ax4.set_xlabel('Normalized x coordinate')
            ax4.set_ylabel('Difference Values')
            ax4.legend()

            # Overall title and layout
            plt.tight_layout()
            plt.suptitle(f"1D Burgers Equation: FNO vs PDE Comparison (ε={epsilon:.4f}, α={datum["alpha"]:.4f}, iter_steps={datum["num_steps"]})\nG: FNO, g: PDE perturbation method: {perturb}", y=1.02, fontsize=14)
            
            # Save and show
            file_folder = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/perturbation_results/1D/plots/"
            os.makedirs(file_folder, exist_ok=True)
            filename = f"{file_folder}burgers_1d_fno_predictions_{perturb}_idx{index+1}_epsilon{epsilon}_numsteps{datum["num_steps"]}_alpha{datum["alpha"]}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(filename)
            plt.close()
            # plt.show()

# Load the pickle file

if __name__ == "__main__":

    # print("Current directory:", os.getcwd())
    folder_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/perturbation_results/1D/"

    # List all files and directories
    extension = ".pkl"  # Example: Filter PNG files
    all_items = [f for f in os.listdir(folder_path) if f.endswith(extension) and "pgdnograd" in f]
    print(all_items)
    for item in all_items[:]:
        if "0.0005" in item:
            with open(f"{folder_path}{item}", "rb") as f:
                data = pickle.load(f)

            perturb = " ".join(item.split("_")[:2])
            
            for index in [0]:
                # for epsilon in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
                for epsilon in [0.1,0.2,0.3,0.4,0.5,0.7,1]:
                    plot_perturb_diff_1d(epsilon, index, data, perturb)