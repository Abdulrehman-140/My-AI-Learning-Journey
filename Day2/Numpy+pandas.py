import numpy as np
import pandas as pd

# make fake marks of 100 students
marks = np.random.randint(30, 100, 100)

df = pd.DataFrame({
    "student": np.arange(1, 101),
    "marks": marks,
})

print(df)
print("Correlation",df.corr(numeric_only=True))
#Broad-casting
matrix = np.array([[1,2,3],
                   [4,5,6]])

vector = np.array([10,20,30])

print("B:",matrix + vector)

#Multiplication
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

C = A @ B      # matrix multiplication
D = A.dot(B)   # same thing

print("\n",C)

#Normalization
x = np.array([10, 20, 30, 40])

normalized = (x - x.mean()) / x.std()
print(normalized)

#softmax
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

print(softmax(np.array([2, 4, 6])))

#Gradient Calcultion
x = np.array([1.0, 2.0, 3.0])
y_pred = 2*x + 1
y_true = np.array([3, 5, 7])

loss = ((y_pred - y_true)**2).mean()
grad = 2*(y_pred - y_true)/len(x)

print("Loss:", loss)
print("Gradient:", grad)

#Encoding into text
df['gender_encoded'] = df['gender'].map({'male': 0, 'female': 1})

#Binning
df['age_group'] = pd.cut(df['age'],
                         bins=[0, 18, 35, 60, 100],
                         labels=['Teen', 'Young Adult', 'Adult', 'Senior'])

#Outlier
q1 = df['marks'].quantile(0.25)
q3 = df['marks'].quantile(0.75)
iqr = q3 - q1

outliers = df[(df['marks'] < q1 - 1.5*iqr) | (df['marks'] > q3 + 1.5*iqr)]

#Feature
df['marks_scaled'] = (df['marks'] - df['marks'].min()) / (df['marks'].max() - df['marks'].min())

#Correlation matrix
print("Correlation",df.corr(numeric_only=True))

df = pd.read_csv("data.csv")

# 1. Drop useless columns
df = df.drop(columns=['id', 'address'], errors='ignore')

# 2. Fill missing numbers
df = df.fillna(df.mean())

# 3. Encode categories
df = pd.get_dummies(df, drop_first=True)

# 4. Scale numeric columns
for col in df.columns:
    if df[col].dtype != 'uint8':   # skip dummy columns
        df[col] = (df[col] - df[col].mean()) / df[col].std()

print(df.head())