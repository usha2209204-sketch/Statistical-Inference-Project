import numpy as np
from scipy import stats
import math

# -----------------------
# Sample Data
# -----------------------

X = np.array([
    0.84, 2.83, 2.37, 3.65, 3.65,
    3.16, 4.30, 1.70, 0.73, 2.79
])

print("Sample Data:")
print(X)

# -----------------------
# Sample Mean
# -----------------------

mean = np.mean(X)
print("\nSample Mean:")
print(mean)

# -----------------------
# Sample Variance
# -----------------------

variance = np.var(X, ddof=1)
print("\nSample Variance:")
print(variance)

# -----------------------
# Sample Standard Deviation
# -----------------------

std = math.sqrt(variance)

print("\nStandard Deviation:")
print(std)

# -----------------------
# Confidence Interval
# -----------------------

alpha = 0.05

n = len(X)

t_value = stats.t.ppf(1-alpha/2, n-1)

margin = t_value * std / math.sqrt(n)

lower = mean - margin
upper = mean + margin

print("\n95% Confidence Interval:")

print(lower, "< μ <", upper)

# -----------------------
# One Sample t-Test
# -----------------------

mu0 = 2.0

t_stat, p_value = stats.ttest_1samp(X, mu0)

print("\nOne Sample t-Test")

print("t-statistic =", t_stat)

print("p-value =", p_value)
