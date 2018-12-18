import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/data_1d.csv")
X = df.iloc[:,0].values
y = df.iloc[:,1].values


plt.scatter(X,y)
plt.show()


den = X.dot(X) - X.mean()*X.sum()
a = (y.dot(X) - y.mean() * X.sum())/den
b = (y.mean() * X.dot(X) - X.mean() * y.dot(X))/den

yhat = a*X + b


plt.scatter(X,y)
plt.plot(X,yhat, color="red")
plt.show()


d1 = y-yhat
d2 = y - y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print("R-squared value is :{}".format(r2))
