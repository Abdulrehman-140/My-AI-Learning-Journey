import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(-10))  # close to 0
print(sigmoid(0))    # 0.5
print(sigmoid(10))   # close to 1

Hours = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
Result = np.array([0,0,0,0,1,1,1,1])  # 0=Fail, 1=Pass

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(Hours, Result)

print(model.predict([[2]]))   # likely fail
print(model.predict([[6]]))   # likely pass
print("Predict Proba:",model.predict_proba([[4.8]]))  # confidence


from sklearn.metrics import accuracy_score

pred = model.predict(Hours)
print("Accuracy:", accuracy_score(Result, pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Result, pred)
print("Confusion Matrix",cm)

import matplotlib.pyplot as plt
plt.scatter(Hours, Result)
plt.plot(Hours, model.predict_proba(Hours)[:,1], color='red') # [:,1] takes only second column. Predicting only pass
plt.xlabel("Study Hours")
plt.ylabel("Pass Probability")
plt.show()