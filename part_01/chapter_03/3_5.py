# Tensor Element Types

import torch

# 3.5.3 Managing the tensor's dtype attribute
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([
    [1, 2],
    [3, 4]
], dtype=torch.short)

print(f"double_points.dtype: {double_points.dtype}")
print(f"short_points.dtype: {short_points.dtype}")


double_points_1 = torch.zeros((10, 2)).double()
short_points_1 = torch.ones(10, 2).short()
print(f"double_points_1: {double_points_1.dtype}")
print(f"short_points_1: {short_points_1.dtype}")

double_points_2 = torch.zeros(10, 2).to(torch.double)
short_points_2 = torch.ones(10, 2).to(dtype=torch.short)
print(f"double_points_2: {double_points_2.dtype}")
print(f"short_points_2: {short_points_2.dtype}")


points_64 = torch.rand(5, dtype=torch.double)
points_short = points_64.to(torch.short)
print(f"points_64 * points_short = {points_64 * points_short}")
