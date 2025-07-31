import os
import pickle
import torch
import numpy as np
from pathlib import Path

def is_torch_file(filepath):
    """Check if file is a PyTorch saved file"""
    with open(filepath, 'rb') as f:
        try:
            # Try reading PyTorch magic number
            magic_number = pickle.load(f)
            return magic_number == 0x1950a86a20  # PyTorch's magic number
        except:
            return False

def needs_conversion(data):
    """Check if data contains PyTorch tensors that need conversion"""
    if isinstance(data, torch.Tensor):
        return True
    elif isinstance(data, dict):
        return any(needs_conversion(v) for v in data.values())
    elif isinstance(data, list):
        return any(needs_conversion(v) for v in data)
    return False

def convert_to_cpu_serializable(data):
    """Recursively convert tensors to numpy arrays only if needed"""
    if isinstance(data, dict):
        return {k: convert_to_cpu_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_cpu_serializable(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    return data

def process_file(filepath):
    """Process a single file with proper error handling"""
    try:
        with open(filepath, 'rb') as f:
            if is_torch_file(filepath):
                data = torch.load(f, map_location='cpu', weights_only=False)
            else:
                data = pickle.load(f)
        
        # Apply Linf slicing if needed
        if "Linf" in filepath.name:
            data = data[180:]
        
        # Only convert if PyTorch tensors are present
        converted = data
        if needs_conversion(data):
            converted = convert_to_cpu_serializable(data)
        
        # Skip processing if no conversion needed and no Linf slicing applied
        if converted is data and "Linf" not in filepath.name:
            print(f"✓ No conversion needed for {filepath.name}")
            return True
        
        # Create backup before overwriting
        backup_path = filepath.with_suffix('.bak')
        os.rename(filepath, backup_path)
        
        # Save converted data
        with open(filepath, 'wb') as f:
            pickle.dump(converted, f, protocol=4)
        
        # Verify the new file
        with open(filepath, 'rb') as f:
            pickle.load(f)  # Test load
            
        os.remove(backup_path)
        return True
    
    except Exception as e:
        print(f"Error processing {filepath.name}: {str(e)}")
        # Restore backup if exists
        if 'backup_path' in locals() and os.path.exists(backup_path):
            os.rename(backup_path, filepath)
        return False

def process_folder(folder_path):
    """Process all .pkl files in folder"""
    folder_path = Path(folder_path)
    success_count = 0
    total_files = 0
    
    for filename in folder_path.glob('*.pkl'):
        total_files += 1
        print(f"Processing {filename.name}...")
        if process_file(filename):
            success_count += 1
    
    print(f"\nConversion complete. Success: {success_count}, Failed: {total_files - success_count}")

if __name__ == "__main__":
    folder_path = "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/perturbation_results/1D"
    process_folder(folder_path)