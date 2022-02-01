import pandas as pd

res = pd.DataFrame()

a = [1,2,3,4,5,6,7,8,9,10]
res.insert(loc=0, column="agent" , value=a)
print(res)