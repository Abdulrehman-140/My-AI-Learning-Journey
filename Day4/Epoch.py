import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

# Initialize weight & bias
w = 0
b = 0

lr = 0.01  # learning rate
epochs = 200

for epoch in range(epochs):
    # Prediction
    y_pred = w*X + b

    # Loss
    loss = np.mean((y_pred - y)**2)

    # Gradients
    dw = (2/len(X)) * np.sum((y_pred - y) * X)
    db = (2/len(X)) * np.sum((y_pred - y))

    # Update
    w -= lr * dw
    b -= lr * db

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print("\nFinal model:")
print("w =", w)
print("b =", b)
