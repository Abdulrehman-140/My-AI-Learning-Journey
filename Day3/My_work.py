import numpy as np
import pandas as pd

#Assignment No. 1
weights = np.array([0.3, 0.5, 0.2])
features = np.array([12, 4, 6])
print(weights.dot(features))

#Assignment No. 2
X = np.array([[2, 3],
              [4, 1],
              [1, 5]])

W = np.array([[0.2, 0.8],
              [0.6, 0.4]])
print(X@W)

#Assignment 3
#f(x) = (x - 5)^2
#df/dx = 2(x-5)
def grad(x):
    return 2 * (x - 5)
x = 0
learning_rate = 0.1
for step in range(5):
    gradient = grad(x)
    x = x - learning_rate * gradient
    print(f"Step {step+1}: x = {x}")

#Assignment 4
y_true = np.array([5, 8, 12])
y_pred = np.array([4, 10, 9])
#(prediction - actual )^2
#-log(correct_probability)
print(np.mean((y_true - y_pred)**2))

#Challenge No.1
w = np.array([0.4, -0.2, 0.1, 0.7])
x = np.array([10, 5, 3, 1])
print(w.dot(x)) #4.0
Normalized = (x - x.mean())/x.std()
print(w.dot(Normalized))

X = np.array([
    [1, 2, 3],
    [4, 0, -1]
])

W = np.array([
    [0.2, -0.5],
    [0.7, 0.1],
    [0.3, 0.8]
])
Z = X@W
A = np.maximum(0,Z)
print(A)

#Relu only Let's positive values. X would be [[1,2,3],[4,0,0]]

xa = 2
h = 0.0001

#Challenge 3 

def f(x):
    return x**2 + 3*x + 2

x = 2
h = 0.0001

grad_num = (f(x+h) - f(x-h)) / (2*h)
print("Numerical gradient:", grad_num)

#Challenge 4
def g(x):
    return 2 * (x - 7)
x = 0
learning_rate = 0.2
for step in range(10):
    gradient = g(x)
    x = x - learning_rate * gradient
    print(f"Step {step+1}: x = {x}")

y_truee = np.array([10, 12, 15, 20])
y_predd = np.array([8, 11, 14, 19])
print(np.mean(y_predd-y_truee)**2)