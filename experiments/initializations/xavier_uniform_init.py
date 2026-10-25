import math
import random

inputs = 2
outputs = 1
weights = 4

r = math.sqrt(2) * math.sqrt(6/(inputs + outputs))
random_number = round(random.uniform(-1 * r, r), 2)

for i in range(weights):
    random_number = round(random.uniform(-1 * r, r), 2)
    print(random_number)