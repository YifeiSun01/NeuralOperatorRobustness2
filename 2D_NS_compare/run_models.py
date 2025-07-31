from FNO2d import FNO2d, RecurrentPredictor
from FNO3d import FNO3d
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image
import os
from tqdm import tqdm
import pickle

def create_heatmap_gif(array1, array2, 
                      title1='Heatmap 1', title2='Heatmap 2', 
                      cmap='viridis', figsize=(18, 6),
                      colorbar=True, 
                      fps=10, dpi=100, text="training", save_path=None):
    # Create temporary directory for frames
    base_path = "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_compare"
    os.makedirs(f'{base_path}/temp_frames', exist_ok=True)
    
    frame_files = []
    
    for idx in range(array1.shape[0]):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot first heatmap
        im1 = ax1.imshow(array1[idx], cmap=cmap, origin='lower')
        ax1.set_title(title1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        if colorbar:
            fig.colorbar(im1, ax=ax1, label='Value')
        
        # Plot second heatmap
        im2 = ax2.imshow(array2[idx], cmap=cmap, origin='lower')
        ax2.set_title(title2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        if colorbar:
            fig.colorbar(im2, ax=ax2, label='Value')

        # Plot second heatmap
        diff = array2[idx]-array1[idx]
        im3 = ax3.imshow(diff, cmap="coolwarm", origin='lower')
        ax3.set_title(f"{title2}-{title1} difference \n mean abs={np.mean(np.abs(diff))}")
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        if colorbar:
            fig.colorbar(im3, ax=ax3, label='Value')
        
        plt.suptitle(f"Navier-Stokes Vorticity Transport (frame {idx+10}) ({text})")
        plt.tight_layout()
        
        # Save frame
        
        frame_file = f"{base_path}/temp_frames/frame_{idx:04d}.png"
        plt.savefig(frame_file, dpi=dpi, bbox_inches='tight')
        frame_files.append(frame_file)
        plt.close()
    
    # Create GIF from frames
    images = [Image.open(f) for f in frame_files]
    parent_folder = os.path.dirname(save_path)
    os.makedirs(parent_folder, exist_ok=True)
    images[0].save(save_path, 
                  save_all=True, 
                  append_images=images[1:], 
                  duration=1000//fps, 
                  loop=0)
    
    # Clean up temporary files
    for f in frame_files:
        os.remove(f)
    os.rmdir(f'{base_path}/temp_frames')
    
    print(f"GIF saved as {text}.gif")

def plot_dual_heatmaps(array1, array2, array3, 
                      title1='Heatmap 1', title2='Heatmap 2', 
                      title3='Heatmap 3', title4="Heatmap 4",
                      cmap='viridis', figsize=(24, 6),
                      vmin=None, vmax=None,
                      colorbar=True, idx=0, text="training",
                      save_path=None, dpi=300):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
    array1 = array1.cpu().numpy()
    array2 = array2.cpu().numpy() 
    array3 = array3.cpu().numpy()

    # Determine value ranges if not specified
    if vmin is None:
        vmin = min(np.nanmin(array2), np.nanmin(array3))
    if vmax is None:
        vmax = max(np.nanmax(array2), np.nanmax(array3))

    # Plot 1: Original data
    im1 = ax1.imshow(array1, cmap=cmap, origin='lower')
    ax1.set_title(title1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    if colorbar:
        fig.colorbar(im1, ax=ax1, label='Value')

    # Plot 2: Prediction/Comparison data
    im2 = ax2.imshow(array2, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax2.set_title(title2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    if colorbar:
        fig.colorbar(im2, ax=ax2, label='Value')

    # Calculate RMSE
    rmse = np.sqrt(np.mean((array2 - array3)**2)).item()

    # Plot 3: Target data with RMSE
    im3 = ax3.imshow(array3, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax3.set_title(f"{title3}\nRMSE={rmse:.4f}")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    if colorbar:
        fig.colorbar(im3, ax=ax3, label='Value')

    # Plot 4: Difference map
    diff = array2 - array3
    im4 = ax4.imshow(diff, cmap="coolwarm", origin='lower')
    ax4.set_title(f"{title4}\nMean Abs Diff={np.mean(np.abs(diff)):.4f}")
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    if colorbar:
        fig.colorbar(im4, ax=ax4, label='Value')
    
    # Main title and layout
    plt.suptitle(f"Navier-Stokes Vorticity Transport idx={idx} ({text})")
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path:
        parent_folder = os.path.dirname(save_path)
        os.makedirs(parent_folder, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Plot saved to {save_path}")
    
    # plt.show()
    plt.close()



base_path = "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO"

models_params = {

    "FNO2d":{
        "initial_to_final":{
            "modes32_width40_epochs500":{
                "model_path":f"{base_path}/2D_NS_old/2D_NS_Zongyi_Li/initial_to_final/saved_models/2D/256_256/modes32_width40_epochs500/input_frame_0_output_frame_19/NS_2d_FNO_model_trainedby_NS_data_zongyi_train.pth",
                "data_path":f"{base_path}/2D_NS_old/2D_NS_Zongyi_Li/initial_to_final/datasets/2D/NS/256_256/output_frame_19/NS_data_zongyi_test.pt",
                "modes":32,
                "width":40,
                },
            "modes64_width60_epochs500":{
                "model_path":f"{base_path}/2D_NS_old/2D_NS_Zongyi_Li/initial_to_final/saved_models/2D/256_256/modes64_width60_epochs500/input_frame_0_output_frame_19/NS_2d_FNO_model_trainedby_NS_data_zongyi_train.pth",
                "data_path":f"{base_path}/2D_NS_old/2D_NS_Zongyi_Li/initial_to_final/datasets/2D/NS/256_256/output_frame_19/NS_data_zongyi_test.pt",
                "modes":64,
                "width":60,
                },
            },

        "recurrent":{
            "modes12_width20_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes12_width20_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":12,
                "width":20,
                "Tin":10,
                "T":10,
                },

            "modes16_width60_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes16_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":16,
                "width":60,
                "Tin":10,
                "T":10,
                },

            "modes32_width40_epochs500_Tin1_T19":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes32_width40_epochs500_Tin1_T19/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":32,
                "width":40,
                "Tin":1,
                "T":19,
                },
            "modes32_width40_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes32_width40_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":32,
                "width":40,
                "Tin":10,
                "T":10,
                },

            "modes32_width60_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes32_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":32,
                "width":60,
                "Tin":10,
                "T":10,
                },

            "modes48_width40_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes48_width40_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":48,
                "width":40,
                "Tin":10,
                "T":10,
                },
            "modes48_width60_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes48_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":48,
                "width":60,
                "Tin":10,
                "T":10,
                },
            
            "modes64_width60_epochs500_Tin1_T19":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes64_width60_epochs500_Tin1_T19/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":64,
                "width":60,
                "Tin":1,
                "T":19,
                },
            "modes64_width60_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes64_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":64,
                "width":60,
                "Tin":10,
                "T":10,
                },
            "modes64_width40_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes64_width40_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":64,
                "width":40,
                "Tin":10,
                "T":10,
                },
            "modes96_width40_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes96_width40_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":96,
                "width":40,
                "Tin":10,
                "T":10,
                },
            "modes128_width40_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes128_width40_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":128,
                "width":40,
                "Tin":10,
                "T":10,
                },
            },
        },

    "FNO3d":{
        "no_padding":{
            "modes1232_modes36_width40_epochs500_Tin10_T10_no_padding":{
                "model_path":f"{base_path}/2D_NS_FNO3d/saved_models/3D/no_padding/modes1232_modes36_width40_epochs500_Tin10_T10/NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes12":32,
                "modes3":6,
                "width":40,
                "Tin":10,
                "T":10,
                },
            "modes1232_modes38_width40_epochs500_Tin1_T19_no_padding":{
                "model_path":f"{base_path}/2D_NS_FNO3d/saved_models/3D/no_padding/modes1232_modes38_width40_epochs500_Tin1_T19/NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes12":32,
                "modes3":8,
                "width":40,
                "Tin":1,
                "T":19,
                },
            "modes1264_modes36_width60_epochs500_Tin10_T10_no_padding":{
                "model_path":f"{base_path}/2D_NS_FNO3d/saved_models/3D/no_padding/modes1264_modes36_width60_epochs500_Tin10_T10/NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes12":64,
                "modes3":6,
                "width":60,
                "Tin":10,
                "T":10,
                },
        },

        "padding":{
            "modes1232_modes38_width40_epochs500_Tin10_T10_padding":{
                "model_path":f"{base_path}/2D_NS_FNO3d/saved_models/3D/padding/modes1232_modes38_width40_epochs500_Tin10_T10/NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes12":32,
                "modes3":8,
                "width":40,
                "Tin":10,
                "T":10,
                "padding":6
                },
            "modes1232_modes312_width40_epochs500_Tin1_T19_padding":{
                "model_path":f"{base_path}/2D_NS_FNO3d/saved_models/3D/padding/modes1232_modes312_width40_epochs500_Tin1_T19/NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes12":32,
                "modes3":12,
                "width":40,
                "Tin":1,
                "T":19,
                "padding":6
                },
            },
        }, 
    }

def predict(models_params, device):
    lower_idx_lim = 0
    upper_idx_lim = 10
    result_dict = {}
    for key1 in models_params.keys():
        result_dict[key1] = {}
        if key1 == "FNO2d": 
            for key2 in models_params["FNO2d"].keys():
                result_dict[key1][key2] = {}
                if key2 == "initial_to_final":  
                    all_models_info_dict = models_params[key1][key2]
                    for key3 in all_models_info_dict:
                        print("========================================================")
                        print(key1,key2,key3)
                        result_dict[key1][key2][key3] = {}
                        info_dict = models_params[key1][key2][key3]
                        print(info_dict)

                        data_path = info_dict["data_path"]
                        modes = info_dict["modes"]
                        width = info_dict["width"]
                        model_path = info_dict["model_path"]

                        data_test = torch.load(
                            data_path, 
                            weights_only=False,
                            map_location=device
                        )
                        model_instance = FNO2d(
                            modes1=modes, 
                            modes2=modes, 
                            width=width,
                            in_channels = 1
                        ).to(device)
                        state_dict = torch.load(
                            model_path,
                            map_location=device
                        )
                        model_instance.load_state_dict(state_dict)
                        model_instance.eval() 

                        with torch.no_grad():
                            for idx in tqdm(range(50)):
                                input_ = data_test["x"][idx]
                                PDE_output = data_test["y"][idx]
                                FNO_output = model_instance(input_.unsqueeze(0).unsqueeze(-1)).squeeze(0).squeeze(-1)

                                error = PDE_output - FNO_output
                                rmse = torch.sqrt(torch.mean(error**2))  # Root Mean Square Error
                                mae = torch.mean(torch.abs(error))      # Mean Absolute Error

                                # Store results
                                result_dict[key1][key2][key3][idx] = {
                                    "RMSE": rmse.item(),       # Convert to Python float
                                    "MAE": mae.item()     # Convert to Python float
                                }
                                """
                                if idx < upper_idx_lim and idx >= lower_idx_lim:
                                    input_frame = 0
                                    output_frame = 19
                                    plot_dual_heatmaps(input_, PDE_output, FNO_output, 
                                        title1=f'input (frame {input_frame})', title2=f'PDE output (frame {output_frame})', 
                                        title3=f'FNO output (frame {output_frame})', title4=f'PDE output - FNO output (frame {output_frame})', 
                                        idx=idx+1, 
                                        text=f"input=frame{input_frame} output=frame{output_frame} modes{modes} width{width} test (initial to final)",
                                        save_path=f"{base_path}/2D_NS_compare/plots/{key1} {key2} {key3} idx{idx}.png")
                                """
                elif key2 == "recurrent":
                    all_models_info_dict = models_params[key1][key2]
                    for key3 in all_models_info_dict:
                        print("========================================================")
                        print(key1,key2,key3)
                        result_dict[key1][key2][key3] = {}
                        info_dict = models_params[key1][key2][key3]
                        print(info_dict)

                        data_path = info_dict["data_path"]
                        modes = info_dict["modes"]
                        width = info_dict["width"]
                        model_path = info_dict["model_path"]
                        Tin = info_dict["Tin"]
                        T = info_dict["T"]

                        data_test = torch.load(
                            data_path, 
                            weights_only=False,
                            map_location=device
                        )
                        model_instance = FNO2d(
                            modes1=modes, 
                            modes2=modes, 
                            width=width,
                            in_channels = Tin
                        ).to(device)
                        state_dict = torch.load(
                            model_path,
                            map_location=device
                        )
                        model_instance.load_state_dict(state_dict)
                        model_instance.eval() 
                        recurrent_model = RecurrentPredictor(model_instance, T_out=T, step=1)

                        for idx in tqdm(range(50)):
                            sample = data_test["y"][idx]  # shape: (x, y, T_total)
                            sample = sample.to(device)

                            # 初始输入 (1, x, y, T_in)
                            current_input = sample[None, ..., :Tin]

                            # 真实输出 (1, x, y, T)
                            ground_truth = sample[None, ..., Tin:Tin+T]

                            # 执行 autoregressive 推理
                            with torch.no_grad():
                                pred = recurrent_model(current_input)  # (1, x, y, T)
                                pred_np = pred.squeeze(0).cpu().numpy().transpose(2, 0, 1)  # (T, x, y)
                                ground_truth_np = ground_truth.squeeze(0).cpu().numpy().transpose(2, 0, 1)  # (T, x, y)
                                input_ = data_test["x"][idx]
                                PDE_output = data_test["y"][idx][...,-2]
                                FNO_output = pred[...,-1].squeeze()

                                error = PDE_output - FNO_output
                                rmse = torch.sqrt(torch.mean(error**2))  # Root Mean Square Error
                                mae = torch.mean(torch.abs(error))      # Mean Absolute Error

                                # Store results
                                result_dict[key1][key2][key3][idx] = {
                                    "RMSE": rmse.item(),       # Convert to Python float
                                    "MAE": mae.item()     # Convert to Python float
                                }
                            """
                            if idx < upper_idx_lim and idx >= lower_idx_lim:
                                input_frame = 0
                                output_frame = 19
                                plot_dual_heatmaps(input_, PDE_output, FNO_output, 
                                        title1=f'input (frame {input_frame})', title2=f'PDE output (frame {output_frame})', 
                                        title3=f'FNO output (frame {output_frame})', title4=f'PDE output - FNO output (frame {output_frame})', 
                                        idx=idx+1, 
                                        text=f"input=frame{input_frame} output=frame{output_frame} modes{modes} width{width} test (recurrent)",
                                        save_path=f"{base_path}/2D_NS_compare/plots/{key1} {key2} {key3} idx{idx}.png")
                                create_heatmap_gif(
                                    ground_truth_np,  # (T, x, y)
                                    pred_np,          # (T, x, y)
                                    title1='PDE Solution',
                                    title2='FNO Prediction',
                                    fps=5,
                                    text=f"modes{model_instance.modes1}-width{model_instance.width} test (Autoregressive T_in={Tin}, T_out={T}) index={idx}",
                                    save_path=f"{base_path}/2D_NS_compare/plots/{key1} {key2} {key3} idx{idx}.gif"
                                )
                            """
                else:
                    pass

        elif key1 == "FNO3d":
            for key2 in models_params["FNO3d"].keys():
                result_dict[key1][key2] = {}
                if key2 == "padding":
                    all_models_info_dict = models_params[key1][key2]
                    for key3 in all_models_info_dict:
                        print("========================================================")
                        print(key1,key2,key3)
                        result_dict[key1][key2][key3] = {}
                        info_dict = models_params[key1][key2][key3]
                        print(info_dict)

                        data_path = info_dict["data_path"]
                        modes12 = info_dict["modes12"]
                        modes3 = info_dict["modes3"]
                        width = info_dict["width"]
                        model_path = info_dict["model_path"]
                        Tin = info_dict["Tin"]
                        T = info_dict["T"]

                        data_test = torch.load(
                            data_path, 
                            weights_only=False,
                            map_location=device
                        )
                        model_instance = FNO3d(
                            modes1=modes12, 
                            modes2=modes12, 
                            modes3=modes3, 
                            width=width, 
                            in_channels=Tin,
                            padding=True).to(device)
                        state_dict = torch.load(
                            model_path, 
                            map_location=device
                        )

                        model_instance.load_state_dict(state_dict)
                        model_instance.eval() 

                        for idx in tqdm(range(50)):
                            # 加载测试数据和模型
                            sample = data_test["y"][idx]  # shape: (x, y, T_total)
                            sample = sample.to(device)

                            # 初始输入 (1, x, y, T_in)
                            current_input = sample[None, ..., :Tin]

                            # 真实输出 (1, x, y, T)
                            ground_truth = sample[None, ..., Tin:Tin+T]

                            # 执行 autoregressive 推理
                            with torch.no_grad():
                                s = 256
                                pred = model_instance(current_input.reshape(1,s,s,1,Tin).repeat([1,1,1,T,1]))  # (1, x, y, T)
                                pred_np = pred.squeeze(0).cpu().numpy().transpose(2, 0, 1, 3).squeeze()  # (T, x, y)
                                ground_truth_np = ground_truth.squeeze(0).cpu().numpy().transpose(2, 0, 1)  # (T, x, y)
                                input_ = data_test["x"][idx]
                                PDE_output = data_test["y"][idx][...,-2]
                                FNO_output = pred[...,-1,0].squeeze()

                                error = PDE_output - FNO_output
                                rmse = torch.sqrt(torch.mean(error**2))  # Root Mean Square Error
                                mae = torch.mean(torch.abs(error))      # Mean Absolute Error

                                # Store results
                                result_dict[key1][key2][key3][idx] = {
                                    "RMSE": rmse.item(),       # Convert to Python float
                                    "MAE": mae.item()     # Convert to Python float
                                }
                            """
                            # 可视化预测和真实值
                            if idx < upper_idx_lim and idx >= lower_idx_lim:
                                input_frame = 0
                                output_frame = 19
                                plot_dual_heatmaps(input_, PDE_output, FNO_output, 
                                        title1=f'input (frame {input_frame})', title2=f'PDE output (frame {output_frame})', 
                                        title3=f'FNO output (frame {output_frame})', title4=f'PDE output - FNO output (frame {output_frame})', 
                                        idx=idx+1, 
                                        text=f"input=frame{input_frame} output=frame{output_frame} modes{modes} width{width} test",
                                        save_path=f"{base_path}/2D_NS_compare/plots/{key1} {key2} {key3} idx{idx}.png")
                                create_heatmap_gif(
                                        ground_truth_np,  # (T, x, y)
                                        pred_np,          # (T, x, y)
                                        title1='PDE Solution',
                                        title2='FNO Prediction',
                                        fps=5,
                                        text=f"modes{model_instance.modes1}-width{model_instance.width} test (Autoregressive T_in={Tin}, T_out={T}) index={idx}",
                                        save_path=f"{base_path}/2D_NS_compare/plots/{key1} {key2} {key3} idx{idx}.gif"
                                    )
                            """
                elif key2 == "no_padding":
                    all_models_info_dict = models_params[key1][key2]
                    for key3 in all_models_info_dict:
                        print("========================================================")
                        print(key1,key2,key3)
                        result_dict[key1][key2][key3] = {}
                        info_dict = models_params[key1][key2][key3]
                        print(info_dict)

                        data_path = info_dict["data_path"]
                        modes12 = info_dict["modes12"]
                        modes3 = info_dict["modes3"]
                        width = info_dict["width"]
                        model_path = info_dict["model_path"]
                        Tin = info_dict["Tin"]
                        T = info_dict["T"]

                        data_test = torch.load(
                            data_path, 
                            weights_only=False,
                            map_location=device
                        )
                        model_instance = FNO3d(
                            modes1=modes12, 
                            modes2=modes12, 
                            modes3=modes3, 
                            width=width, 
                            in_channels=Tin,
                            padding=False).to(device)
                        state_dict = torch.load(
                            model_path, 
                            map_location=device
                        )

                        model_instance.load_state_dict(state_dict)
                        model_instance.eval() 

                        for idx in tqdm(range(50)):
                            # 加载测试数据和模型
                            sample = data_test["y"][idx]  # shape: (x, y, T_total)
                            sample = sample.to(device)

                            # 初始输入 (1, x, y, T_in)
                            current_input = sample[None, ..., :Tin]

                            # 真实输出 (1, x, y, T)
                            ground_truth = sample[None, ..., Tin:Tin+T]

                            # 执行 autoregressive 推理
                            with torch.no_grad():
                                s = 256
                                pred = model_instance(current_input.reshape(1,s,s,1,Tin).repeat([1,1,1,T,1]))  # (1, x, y, T)
                                pred_np = pred.squeeze(0).cpu().numpy().transpose(2, 0, 1, 3).squeeze()  # (T, x, y)
                                ground_truth_np = ground_truth.squeeze(0).cpu().numpy().transpose(2, 0, 1)  # (T, x, y)
                                input_ = data_test["x"][idx]
                                PDE_output = data_test["y"][idx][...,-2]
                                FNO_output = pred[...,-1,0].squeeze()

                                error = PDE_output - FNO_output
                                rmse = torch.sqrt(torch.mean(error**2))  # Root Mean Square Error
                                mae = torch.mean(torch.abs(error))      # Mean Absolute Error

                                # Store results
                                result_dict[key1][key2][key3][idx] = {
                                    "RMSE": rmse.item(),       # Convert to Python float
                                    "MAE": mae.item()     # Convert to Python float
                                }
                            """
                            # 可视化预测和真实值
                            if idx < upper_idx_lim and idx >= lower_idx_lim:
                                input_frame = 0
                                output_frame = 19
                                plot_dual_heatmaps(input_, PDE_output, FNO_output, 
                                        title1=f'input (frame {input_frame})', title2=f'PDE output (frame {output_frame})', 
                                        title3=f'FNO output (frame {output_frame})', title4=f'PDE output - FNO output (frame {output_frame})', 
                                        idx=idx+1, 
                                        text=f"input=frame{input_frame} output=frame{output_frame} modes{modes} width{width} test",
                                        save_path=f"{base_path}/2D_NS_compare/plots/{key1} {key2} {key3} idx{idx}.png")
                                create_heatmap_gif(
                                        ground_truth_np,  # (T, x, y)
                                        pred_np,          # (T, x, y)
                                        title1='PDE Solution',
                                        title2='FNO Prediction',
                                        fps=5,
                                        text=f"modes{model_instance.modes1}-width{model_instance.width} test (Autoregressive T_in={Tin}, T_out={T}) index={idx}",
                                        save_path=f"{base_path}/2D_NS_compare/plots/{key1} {key2} {key3} idx{idx}.gif"
                                    )
                            """

                else:
                    pass

        else:
            pass

    return result_dict

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_dict = predict(models_params, device)
    with open(f'{base_path}/2D_NS_compare/results.pkl', 'wb') as f:
        pickle.dump(result_dict, f)



