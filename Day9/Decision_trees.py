from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.random.randn(100, 1)
y = (X[:,0] > 0).astype(int)

#Adding noise
noise = np.random.randint(0, 2, size=100)
y = np.logical_xor(y, noise).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier(max_depth=20)

model.fit(X_train, y_train)

print("Train Accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))


#Challenge 1: The data is too clean to demonstrate overfitting and underfitting before adding noise. Train and Test accuracy both are 1.0
#After adding noise, Things become complicated
