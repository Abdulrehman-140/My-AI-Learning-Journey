import numpy as np
from sklearn.model_selection import train_test_split #This train_test_split divides the data into training and data sets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

Area = np.array([[800], [1000], [1500], [2000], [2500], [3000], [3500]])
Price = np.array([90, 120, 150, 180, 220, 260, 300])
A_train, A_test, P_train, P_test = train_test_split(  #Area train and test, Price train and test
    Area, Price, test_size=0.3, random_state=42    #0.3 means 30% of data to be allocated for testing set and 70% for training
) #Random state sets seed for random number generation
model = LinearRegression()
model.fit(A_train, P_train)

print("Weight:", model.coef_)
print("Bias:", model.intercept_)

P_pred = model.predict(A_test)
print("Predicted:", P_pred)
print("Actual:", P_test)

mse = mean_squared_error(P_test, P_pred)
r2 = r2_score(P_test, P_pred)

print("MSE:", mse)
print("R² Score:", r2)

plt.scatter(Area, Price, label='Actual Data')
plt.plot(Area, model.predict(Area), label='Model Prediction')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.legend()
plt.show()
