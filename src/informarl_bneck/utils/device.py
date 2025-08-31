"""
Device management for server GPU usage
"""
import torch
import os


def get_device(gpu_id=None, force_cpu=False):
    """
    Get the best available device for server usage
    
    Args:
        gpu_id: Specific GPU ID to use (0, 1, 2, etc.)
        force_cpu: Force CPU usage even if GPU available
    """
    if force_cpu:
        return torch.device("cpu")
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu")
    
    # Get GPU count and info
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    
    # Select GPU
    if gpu_id is not None:
        if gpu_id >= gpu_count:
            print(f"GPU {gpu_id} not available, using GPU 0")
            gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU {gpu_id}")
    else:
        # Auto-select best GPU (most memory)
        best_gpu = 0
        max_memory = 0
        
        for i in range(gpu_count):
            torch.cuda.set_device(i)
            memory_free = torch.cuda.get_device_properties(i).total_memory
            if memory_free > max_memory:
                max_memory = memory_free
                best_gpu = i
        
        device = torch.device(f"cuda:{best_gpu}")
        print(f"Auto-selected GPU {best_gpu} (most memory)")
    
    # Set memory allocation strategy for server efficiency
    torch.cuda.empty_cache()  # Clear cache
    
    return device


def setup_gpu_environment():
    """Setup optimal GPU environment for server training"""
    
    # Environment variables for better server performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async GPU operations
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Use cuDNN v8
    
    if torch.cuda.is_available():
        # Memory management
        torch.backends.cudnn.benchmark = True  # Optimize cuDNN
        torch.backends.cudnn.deterministic = False  # Faster but less deterministic
        
        # Memory allocation strategy
        torch.cuda.empty_cache()
        
        print("GPU environment optimized for server training")
        
        # Print memory info
        for i in range(torch.cuda.device_count()):
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i} Memory: {memory_allocated:.1f}GB used, {memory_cached:.1f}GB cached, {memory_total:.1f}GB total")


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory cache cleared")


def get_memory_usage():
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return "CPU mode"
    
    current_device = torch.cuda.current_device()
    memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
    memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3
    memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
    
    return f"GPU {current_device}: {memory_allocated:.1f}GB/{memory_total:.1f}GB ({memory_allocated/memory_total*100:.1f}%)"