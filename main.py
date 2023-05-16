import torch
import numpy as np

# Initializing a tensor

# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# or from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# even from another tensor!
x_ones = torch.ones_like(x_data) # retains property of the argument tensor unless specified otherwise
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the data type of x_data
print(f"Random Tensor: \n {x_rand} \n")
print(x_data.device, x_np.device, x_ones.device, x_rand.device)
if torch.cuda.is_available():
    x_data = x_data.to("cuda")

print(x_data.device, x_np.device, x_ones.device, x_rand.device)