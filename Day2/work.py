import numpy as np
import pandas as pd

#Assignment 1
logits = np.array([3.2, 5.1, 1.7])
i = np.exp(logits-logits.max())
print(i/i.sum())
For_One = [0.05,0.9,0.05]
Loss = -np.log(0.9)
print(f"Loss:{Loss}")

#Assignment 2
Mat = np.random.randint(1,36,(6,6))
print(Mat)
Normalized = (Mat -Mat.mean())/Mat.std()
print(Normalized)
print(Mat.T @ Normalized)
#Normalization is important for ML because Models freak out if one row has smaller values and the other has giant values.

#Assignment 3
Data = {'Name':['Ali','Umar','Abc'],'Age':[23,23,25],'Score':[87,92,89],'City':['xyz','xyzz','xyz'],'Gender':['male','male','female']}
Data = pd.DataFrame(Data)
print(Data)
Data['Age'] = Data['Age'].fillna(Data['Age'].mean())
Data['Gender'] = Data['Gender'].map({'male':0,'female':1})
Hot_coded = pd.get_dummies(Data['City'])
print(Hot_coded)

Standardized= (Data['Score']-Data['Score'].mean())/Data['Score'].std()
print(Standardized)

Data['AgeGroup'] = pd.cut(
    Data['Age'],
    bins=[0, 19, 35, 59, 100], 
    labels=['Teen', 'Young', 'Adult', 'Old']
)
print(Data.corr(numeric_only=True))

#Outliers detection
q1 = Data['Score'].quantile(0.25)
q3 = Data['Score'].quantile(0.75)
iqr = q3 - q1

outliers = Data[(Data['Score'] < q1 - 1.5*iqr) | (Data['Score'] > q3 + 1.5*iqr)]
print(outliers,outliers.count())

#Pipeline, Same but different data only for this purpose 😁
Data2 = pd.DataFrame({
    'Name':['Ali','Umar','Abc'],
    'Age':[23,23,25],
    'Score':[87,92,89],
    'City':['xyz','xyzz','xyz'],
    'Gender':['male','male','female']
})

# 1. Drop useless column
Data2 = Data2.drop(columns=['Name'])

# 2. Fill missing values
Data2['Age'] = Data2['Age'].fillna(Data2['Age'].mean())

# 3. Encode gender
Data2['Gender'] = Data2['Gender'].map({'male':0, 'female':1})

# 4. One-hot encode city
hot = pd.get_dummies(Data2['City'])
Data2 = pd.concat([Data2.drop('City',axis=1), hot], axis=1)

# 5. Standardize scores
Data2['Score'] = (Data2['Score'] - Data2['Score'].mean()) / Data2['Score'].std()

print(Data2.head())
