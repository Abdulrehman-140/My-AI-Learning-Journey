import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

X = np.random.randn(200, 5)   # 5 features
true_w = np.array([2, 0, 0, 0, 0])  # Only ONE feature matters
y = (X @ true_w + np.random.randn(200)*1.5 > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model_l1 = LogisticRegression(penalty='l1',solver='saga', max_iter=5000, C=0.1)
model_l1.fit(X_train, y_train)

print("Train:", accuracy_score(y_train, model_l1.predict(X_train)))
print("Test :", accuracy_score(y_test, model_l1.predict(X_test)))
print("Weights:", model_l1.coef_)
