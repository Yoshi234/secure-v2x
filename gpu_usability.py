import time
import os
import torch

def run_on_gpu(devices):
    # Set the active device for each GPU in the devices list
    for device in devices:
        torch.cuda.set_device(device)
        print(f"Running on GPU {device}")

    # Create a tensor on GPU
    tensor = torch.tensor([1.0]).cuda()

    # Perform a computation repeatedly for 15 seconds
    start_time = time.time()
    while time.time() - start_time < 15:
        result = tensor * 2

    # Synchronize and measure the elapsed time
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    print(f"Computation time: {elapsed_time:.2f} seconds")

def set_visible_gpus(gpu_indices):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(index) for index in gpu_indices)

if __name__ == '__main__':
    # Specify the indices of the GPUs you want to use
    gpu_indices = [0]

    # Set the CUDA_VISIBLE_DEVICES environment variable
    set_visible_gpus(gpu_indices)

    # Call the function to run on GPU
    run_on_gpu(gpu_indices)
    
# import tensorflow as tf

# def run_on_gpus(gpu_indices):
#     # Set the visible GPUs
#     visible_devices = [f"/device:GPU:{idx}" for idx in gpu_indices]
#     tf.config.set_visible_devices(visible_devices, 'GPU')

#     # Your code here...
#     # Perform TensorFlow computations on the specified GPUs

# if __name__ == '__main__':
#     # Specify the indices of the GPUs you want to use
#     gpu_indices = [0, 2, 3]

#     # Call the function to run on GPUs
#     run_on_gpus(gpu_indices)