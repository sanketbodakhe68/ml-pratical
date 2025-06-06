import numpy as nu
import statistics

# Sample data
data = [12, 15, 12, 18, 19, 15, 12, 14, 17, 19 ,20, 45, 34, 100]

# Compute mean
mean = statistics.mean(data)
print(f"Mean: {mean}")

# Compute median
median = statistics.median(data)
print(f"Median: {median}")

# Compute mode
try:
    mode = statistics.mode(data)
    print(f"Mode: {mode}")
except statistics.StatisticsError:
    print("Mode: No unique mode found (multiple modes present).")

# Compute variance
variance = statistics.variance(data)
print("Variance: {variance}")

# Compute standard deviation
std_deviation = statistics.stdev(data)
print(f"Standard Deviation: {std_deviation}")
