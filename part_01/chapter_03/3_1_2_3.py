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

# -------------------------------------------------------------------

# 3.2.3  the essence of tensors

# creating appropriately sized array
points = torch.zeros(6)

# assigning values to it
points[0] = 4.0
points[1] = 1.0
points[2] = 5.0
points[3] = 3.0
points[4] = 2.0
points[5] = 1.0

print(f"points: {points}")

# passing python list to the constructor to the same effect
points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
print(f"points: {points}")


# getting the coordinates of the first point
print(f"coordinates: {float(points[0]),float(points[1])}")


# 2D tensor
points = torch.tensor([
    [4.0, 1.0],
    [5.0, 3.0],
    [2.0, 1.0]
])

print(f"2d points: {points}")

# tensor shape
print(f"shape of tensor: {points.shape}")


# using zeros and ones to initialize the tensor
points = torch.zeros(3, 2)
print(f"points: {points}")

points = torch.tensor([
    [4.0, 1.0],
    [5.0, 3.0],
    [2.0, 1.0]
])

print(f"2d tensor: {points}")
print(f"points[0, 1]: {points[0, 1]}")
print(f"y coordinates of zeroth point: {points[0]}")

# -------------------------------------------------------------------

# 3.3  indexing tensors

# all rows after the first, implicitly all columns
print(f"points[1:]: {points[1:]}")

# all rows after the first, first column
print(f"points[1:, :] : {points[1:, :]}")

# all rows after the first, first column
print(f"points[1:, 0] : {points[1:, 0]}")

# adds a dimension of size 1, just like unsqueeze
points = points[None]
print(points)
