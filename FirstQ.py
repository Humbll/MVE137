import random

# Define the probabilities and number of samples
population_percentage_infected = 0.02  # 2% of the population is infected
false_negative_rate = 0.999  # Probability of a true positive if infected
false_positive_rate = 0.005  # Probability of a false positive if not infected
num_samples = 10**4  # Number of samples

# Generate random infection status for the population
infection_status = [1 if random.random() < population_percentage_infected else 0 for _ in range(num_samples)]

# Simulate the virus test results for each individual
test_results = []
for status in infection_status:
    if status == 1:
        # Infected individual
        if random.random() < false_negative_rate:
            test_results.append(1)  # Positive result
        else:
            test_results.append(0)  # Negative result
    else:
        # Non-infected individual
        if random.random() < false_positive_rate:
            test_results.append(1)  # False positive result
        else:
            test_results.append(0)  # True negative result

# Calculate the probability that a person has the virus given a positive test result
num_positive_tests = sum(test_results)
num_true_positives = sum([1 for i in range(num_samples) if test_results[i] == 1 and infection_status[i] == 1])
probability_infected_given_positive = num_true_positives / num_positive_tests

print("Probability that a person has the virus given a positive test result:", probability_infected_given_positive)