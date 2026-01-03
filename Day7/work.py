from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

Hours_and_Sleep = np.array([[1,5],[2,6],[3,5],[4,7],[5,6],[6,8],[7,7]])
Pass = np.array([0,0,0,0,1,1,1])
model = LogisticRegression()
model.fit(Hours_and_Sleep,Pass)
pred = model.predict(Hours_and_Sleep)
print("Precision:", precision_score(Pass, pred))
print("Recall:", recall_score(Pass, pred))
print("F1:", f1_score(Pass, pred))