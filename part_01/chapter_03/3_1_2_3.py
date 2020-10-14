# 3.2.1  from python lists to pytorch tensors

a = [1.0, 2.0, 3.0]

# first element of list
print(f"first element: {a[0]}")

# second element of list
print(f"second element: {a[1]}")

# third element of list
print(f"third element: {a[2]}")

# all elements of list
print(f"all elements: {a}")

# -------------------------------------------------------------------

# 3.2.2  constructing our first tensors

# importing the torch module
import torch

# creating a one-dimensional tensor of size 3 filled by 1s
a = torch.ones(3)
print(f"a: {a}")

# second element of tensor
print(f"second element of tensor: {a[1]}")

# converting to float
print(f"to float: {float(a[1])}")

# assigning a new value to element of tensor
a[2] = 2.0
print(f"all elements of tensor: {a}")
