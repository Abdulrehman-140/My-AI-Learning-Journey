import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,precision_score,accuracy_score

Sleep = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
Health = np.array([0,0,0,0,0,1,1,1,1,0]) #1 for healthy, 0 for not

model = LogisticRegression()
model.fit(Sleep,Health)

probs = model.predict_proba(Sleep)[:,1]
Normal_preds = (probs>0.5).astype(int)
generous_preds = (probs>0.3).astype(int)
strict_preds = (probs>0.7).astype(int)

print("Strict (0.8):", strict_preds)
print("Generous (0.1):", generous_preds)

print("Generous Precision Score:",precision_score(Health,generous_preds))
print("Recall Score:",recall_score(Health,generous_preds)) 

print("Strict Precision Score:",precision_score(Health,strict_preds))
print("Recall Score:",recall_score(Health,strict_preds))

from sklearn.metrics import confusion_matrix
print("Generous CM:\n", confusion_matrix(Health, generous_preds))
print("Strict CM:\n", confusion_matrix(Health, strict_preds))


Exam_score = np.array([[98],[89],[67],[78],[45],[39],[56],[78],[93],[12]])
Pass = np.array([1,1,1,1,1,0,1,1,1,1])
model = LogisticRegression()
model.fit(Exam_score,Pass)
pred = model.predict(Exam_score)
print(accuracy_score(Pass,pred))

#Model has 90% accuracy and It needs higher precision not to say Pass for everyone.

#Recall for Airport Screening to avoid any unpleasant event and for Cancer Screening to make sure That each cancer patient is treated in time. Precision for loan approval to make sure if That person even got a plan or not


#Visualizing Trade-off
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(Health, probs)

plt.plot(recall, precision, color='blue', label='PR Curve')
plt.xlabel('Recall (Generosity)')
plt.ylabel('Precision (Strictness)')
plt.title('Precision-Recall Tradeoff')
plt.show()

#ROC and AUC
from sklearn.metrics import roc_curve, roc_auc_score

# Calculate the ROC curve points
fpr, tpr, thresholds_roc = roc_curve(Health, probs)

# Calculate the AUC (the single number score)
auc_score = roc_auc_score(Health, probs)

print(f"The model's overall score (AUC) is: {auc_score}")

# Plotting the ROC Curve
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--') # The "Random Guess" line
plt.xlabel('False Positive Rate (False Alarms)')
plt.ylabel('True Positive Rate (Correct Detections)')
plt.legend()
plt.show()