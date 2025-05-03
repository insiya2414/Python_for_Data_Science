'''Q31) Simulate a biased coin flip given a fair coin function'''
# Core Idea (Von Neumann’s Method)
# When using a fair coin (equal chance of heads or tails), we can simulate a biased outcome by eliminating the bias from patterns like:
# Heads-Tails → Return 0
# Tails-Heads → Return 1
# Heads-Heads or Tails-Tails → Repeat the process
# This ensures equal probability for 0 and 1.

import random

def biased_coin():
    flip1, flip2 = random.randint(0, 1), random.randint(0, 1)  # Fair coin: 0 = tails, 1 = heads
    if flip1 != flip2:
        return flip1  # Return either 0 or 1, depending on the first flip
    return biased_coin()  # If both flips are same, try again
# Output: 0 or 1

'''Q32) Calculate the confidence interval for a given dataset (assume normal distribution)'''
# A confidence interval gives you a range of values in which you're fairly confident the true population mean lies.
# We usually say something like:
# “We are 95% confident that the true mean lies between X and Y.”
# Formula: CI = x(mean) +- z(s/sqrt(n)) [x= sample mean, s= sample standard deviation, n = numberof samples, z =z score]
import numpy as np
from scipy.stats import norm

def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)                      # Step 1: calculate sample mean
    std = np.std(data, ddof=1)                # Step 2: sample standard deviation (ddof=1 = sample std)
    z = norm.ppf((1 + confidence) / 2)        # Step 3: z-score for given confidence (e.g. 1.96 for 95%)
    margin_of_error = z * (std / np.sqrt(len(data)))  # Step 4: margin of error(calculation for everything after +-)
    return mean - margin_of_error, mean + margin_of_error  # Step 5: return confidence interval

'''Q33)  Implement the Chi-squared test for independence on a contingency table'''
# Chi-Squared Test for Independence to find out if two categorical variables are related.
# Contingency_table — a 2D list (like a matrix) representing observed values from your study.
# Term	   |    Meaning
# chi2	   |    Test statistic — larger = more difference between groups
# p	       |    Probability of seeing this result by chance [Low p (< 0.05)	Reject independence — likely related, 
#          |    High p (≥ 0.05)	Fail to reject — likely independent]
# dof	   |    Degrees of freedom — needed for reference tables (rows-1)x(columns-1)
# expected |	What the values should be if the groups were independent

import numpy as np
from scipy.stats import chi2_contingency

def chi_squared_test(contingency_table):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p

# Contingency table
table = [[10, 20], 
         [20, 40]]

print(chi_squared_test(table))
# Output: (0.0, 1.0) x^2 = 0.0: This means observed = expected values exactly, p-value = 1.0: The higher the p-value, the more likely the variables are independent. 

'''Q34) Generate random numbers following a given probability distribution'''
import numpy as np

def generate_random_numbers(elements, probabilities, size):
    return np.random.choice(elements, size=size, p=probabilities)


print(generate_random_numbers([1, 2, 3], [0.2, 0.5, 0.3], 10))
# Output: [2 1 2 2 2 3 1 2 2 2]

'''Q35) Implement k-nearest neighbors from scratch'''
#  What is k-Nearest Neighbors (k-NN)?
# It’s a simple, instance-based learning algorithm:
# Given a test point, it finds the k closest training examples (based on distance).
# It predicts the class (for classification) or average value (for regression) based on those neighbors.

# Basic Steps:
# Calculate the distance from the test point to all training points (typically Euclidean).
# Sort the distances and pick the k smallest.
# Vote (classification) or average (regression) the results of the neighbors.

import numpy as np
from collections import Counter

def knn(X_train, y_train, X_test, k):
    # Calculate Euclidean distance between each training point and test point
    distances = [np.linalg.norm(x - X_test) for x in X_train]
    # Get indices of k nearest neighbors and their corresponding labels
    k_neighbors = [y_train[i] for i in np.argsort(distances)[:k]]
    # Return most common class label among k neighbors
    return Counter(k_neighbors).most_common(1)[0][0]

X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = [0, 1, 1]
X_test = np.array([2.5, 3])
print(knn(X_train, y_train, X_test, 2))
# Output: 1