import matplotlib.pyplot as plt
import math

def calculate_best_k(p):
    # Approximation for the best value of k when p is small
    # For example, you can use a rule of thumb like k = ceil(1/p)
    return math.ceil(1 / p)

# Step 1: Derive an approximation for the best value of k when p is small
# You'll need to define your mathematical model here.

# Step 2: Test if pooling is better for different values of p
p_values = [0.01, 0.02, 0.03, ..., 0.5]  # Replace ellipsis with the actual list of p values
n = 102  # Total number of individuals

# Initialize lists to store results
individual_tests = []
pooling_tests = []

for p in p_values:
    # Calculate the expected number of tests for individual testing
    individual_tests.append(n * (1 - (1 - p) ** n))

    # Calculate the expected number of tests for pooling using the best k
    k = calculate_best_k(p)  # Replace with your function for k approximation
    pooling_tests.append(n * (1 - (1 - p/k) ** (n/k)))

# Step 3: Plot the results
plt.plot(p_values, individual_tests, label='Individual Testing')
plt.plot(p_values, pooling_tests, label='Pooling')
plt.xlabel('p')
plt.ylabel('Expected Number of Tests')
plt.legend()
plt.show()