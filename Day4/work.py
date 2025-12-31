from sklearn.linear_model import LinearRegression
import numpy as np

Area = np.array([[1000], [1500], [2000], [2500], [3000]])
Price = np.array([120, 150, 200, 230, 280])

model = LinearRegression()
model.fit(Area,Price)
print(model.coef_)
print(model.intercept_)
print(model.predict([[2800]]))

X = np.array([[1, 10],
     [2, 20],
     [3, 30],
     [4, 40]])
y = np.array([50, 100, 150, 200])
model.fit(X,y)
print(model.coef_)
print(model.predict([[5,50]]))


A = np.random.randint(1,10,(1,6))
B = np.random.randint(1,10,(1,6))

w = 0
b = 0
l_rate = 0.01
Epochs=200
for epoch in range(Epochs):
   B_pred = w*A + b
   loss = np.mean((B_pred-B)**2)

   dw = (2/len(A)) * np.sum((B_pred - B) * A)
   db = (2/len(A)) * np.sum((B_pred - B))

   w= w-l_rate*dw 
   b= b-l_rate*db
   if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
print(f"Final Model\n Weight:{w},Bias:{b}")
