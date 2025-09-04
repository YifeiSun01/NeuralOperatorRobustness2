import matplotlib.pyplot as plt



def visualize_GRF(arr):
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    
    plt.figure(figsize=(8, 6))
    
    if arr.ndim == 1:
        # 1D array - line plot
        plt.plot(np.arange(len(arr)), arr)
        plt.title('1D Array - Line Plot')
        plt.xlabel('Index')
        plt.ylabel('Value')
        
    elif arr.ndim == 2:
        # 2D array - heatmap
        plt.imshow(arr, cmap='viridis', aspect='auto')
        plt.colorbar(label='Value')
        plt.title('2D Array - Heatmap')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        
    else:
        raise ValueError("Array must be either 1D or 2D")
    
    plt.tight_layout()
    plt.show()