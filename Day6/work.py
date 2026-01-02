import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

Age = np.array([[18],[22],[25],[30],[35],[40]])
Buy = np.array([0,0,1,1,1,1])

model = LogisticRegression()
model.fit(Age,Buy)
print(model.predict([[28]]))
print(model.predict_proba([[28]]))


Hours_and_Sleep = np.array([[1,5],[2,6],[3,5],[4,7],[5,6],[6,8],[7,7]])
Pass = np.array([0,0,0,0,1,1,1])
model = LogisticRegression()
model.fit(Hours_and_Sleep,Pass)
print(model.predict([[5,8]]))
pred = model.predict(Hours_and_Sleep)
print(accuracy_score(Pass,pred))
print(confusion_matrix(Pass,pred)) #Returns a  2D matrix with [[Truely Fail, predicted pass], [FN, Truely pass]].

# In regression, We predict how much or we predict a number figure. . In classification, Just 0 or 1. For example: In regression: Rate of house over location. In classification: If the house would be sold or not?
# Sigmoid is the backbone of Logistic regression which gives probalities between 1 or 0, no matter how big number we give to it
# Means I hoped for 'yes', But she said 'No' 
