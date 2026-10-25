import math
import numpy as np

# Values to plug in.
mean = 0.5
input_connections = 2
output_connections = 1
weights = 4

# Calculating standard deviation.
std_dev = math.sqrt(2) * (math.sqrt(2/(input_connections + output_connections)))

# Generate random samples from a normal distribution
data = np.random.normal(mean, std_dev**2, weights)

for data_point in data:
    print(round(data_point, 2))