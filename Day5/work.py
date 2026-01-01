import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

Hours = np.array([[1],[2],[3],[4],[5],[6],[7]])
Score = [50,55,60,70,75,80,88]

h_train,h_test,s_train,s_test = train_test_split(
    Hours, Score, test_size=0.2,random_state=42
)

model = LinearRegression()
model.fit(h_train,s_train)
print(model.predict([[6.5]]))