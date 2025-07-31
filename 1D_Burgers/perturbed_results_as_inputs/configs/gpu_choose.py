import torch

def select_optimal_gpu():
    if not torch.cuda.is_available():
        return torch.device('cpu')

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return torch.device('cpu')

    # Get current memory usage and select GPU with the most free memory
    free_memories = []
    for i in range(num_gpus):
        torch.cuda.set_device(i)  # Switch to GPU i temporarily
        allocated = torch.cuda.memory_allocated(i)  # Current GPU memory allocated (in bytes)
        reserved = torch.cuda.memory_reserved(i)    # Current GPU memory reserved (in bytes)
        free_memory = reserved - allocated          # Approximate free memory
        free_memories.append(free_memory)

    # Select GPU with the most free memory (smallest current usage)
    optimal_gpu = free_memories.index(max(free_memories))
    print(f"Selected GPU {optimal_gpu} ({torch.cuda.get_device_name(optimal_gpu)}) with {free_memories[optimal_gpu] / 1e9:.2f} GB free memory")

    return torch.device(f'cuda:{optimal_gpu}')