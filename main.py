#Tensors are like multi-dimensional arrays
#There are over 100 tensor operations.
import torch
import numpy as np

#Initializing a tensor directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

#Creating a tensor from a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#Creating a tensor from another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#Displaying the attributes of a tensor
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#You can run operations on GPU if available. GPU is typically faster than cpu
#We can move tensor to GPU using the .to method
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

#We can do operations, operations are very similar to NumPy API

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)