import numpy as np
import pandas as pd

# --------------------------
# Challenge 1
# --------------------------

w = np.array([0.4, -0.2, 0.1, 0.7])
x = np.array([10, 5, 3, 1])

dot1 = w.dot(x)
# Manual:
# (0.4*10) + (-0.2*5) + (0.1*3) + (0.7*1)

x_norm = (x - x.mean()) / x.std()
dot2 = w.dot(x_norm)

print("Original dot:", dot1)
print("After normalization:", dot2)

# Explain in comments:
# 1. Why dot value changed
# 2. Why normalization matters


# --------------------------
# Challenge 2
# --------------------------

X = np.array([[1, 2, 3],
              [4, 0, -1]])

W = np.array([[0.2, -0.5],
              [0.7, 0.1],
              [0.3, 0.8]])

Z = X @ W

# ReLU manually
A = np.maximum(0, Z)

print("Z =\n", Z)
print("ReLU(Z) =\n", A)

# Explain in comments:
# ReLU removes negative values, keeps positive ones.


# --------------------------
# Challenge 3
# --------------------------

def f(x):
    return x**2 + 3*x + 2

x = 2
h = 0.0001

grad_num = (f(x+h) - f(x-h)) / (2*h)
print("Numerical gradient:", grad_num)


# --------------------------
# Challenge 4
# --------------------------

def g_prime(x):
    return 2*(x-7)

x = 0
lr = 0.2

for step in range(10):
    grad = g_prime(x)
    x = x - lr * grad
    print(f"Step {step+1}: x = {x}")

# Comment what happens to x:
# - How it moves toward 7
# - Why movements shrink over time


# --------------------------
# Challenge 5
# --------------------------

y_true = np.array([10, 12, 15, 20])
y_pred = np.array([8, 11, 14, 19])

def my_mse(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)

print("My MSE:", my_mse(y_true, y_pred))
print("Numpy MSE:", np.mean((y_pred - y_true)**2))
