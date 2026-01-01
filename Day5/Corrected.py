import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

Hours = np.array([[1],[2],[3],[4],[5],[6],[7]])
Score = np.array([50,55,65,70,75,80,88])

X_train, X_test, y_train, y_test = train_test_split(
    Hours, Score, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Prediction for 6.5 hours:", model.predict([[6.5]]))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


plt.scatter(Hours, Score, label="Actual")
plt.plot(Hours, model.predict(Hours), color='red', label="Model")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.legend()
plt.show()

Score_2 = np.array([[48],[52],[60],[68],[72],[80],[90]])
HoursandSleep = np.array([[1,6],[2,7],[3,6],[4,7],[5,8],[6,8],[7,9]])
model = LinearRegression()
model.fit(HoursandSleep, Score_2)
print(model.predict([[5,8]]))




X = np.random.randint(1,50,(50,1))
y = 2*X.flatten() + 10 + np.random.randn(50)*5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, color='red', label="Predicted")
plt.legend()
plt.show()
