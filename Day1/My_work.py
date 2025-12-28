import numpy as np
import pandas as pd

arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
rand = np.random.randn(5,5)
print(arr)
print(rand)

ray1 = [5,1,3]
ray2 = [2,4,6]
print(np.dot(ray1,ray2))

random = np.random.rand(5,3)
print(random.reshape(3,5))

Data = {'Name':['Ali','Umar','ab'],'Age':[12,13,14],'Score':[69,76,59],'Gender':['Male','Custom','Female']}
Df = pd.DataFrame(Data)
print(Df)
print("Groups:",Df.groupby('Gender').count())
print(Df.sort_values('Age',ascending=False))
print(Df.isnull().sum)

#L2 norm of [6,8] is 10
# 9x^2
#It means strong negative relationship