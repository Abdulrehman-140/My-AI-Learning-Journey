from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([3,5,7,9,11])

model = LinearRegression()
model.fit(X, y)

print("Weight:", model.coef_)
print("Bias:", model.intercept_)
print("Predicted:", model.predict([[6]]))
