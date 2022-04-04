#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df0 = pd.read_csv("Ecommerce_Customers.csv")
columns = ["Email","Address","Avatar","Avg Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent"]
df0 = pd.read_table("Ecommerce_Customers.csv", sep=',', header=None, names=columns)
X = df0[['Avg Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = df0['Yearly Amount Spent']


# In[11]:


df0


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.4 ,random_state=42)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
Predict = reg.predict( X_test)


# In[38]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test,Predict))
print('MSE:', metrics.mean_squared_error(y_test,Predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,Predict)))


# In[39]:


plt.scatter(y_test,Predict)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[43]:


x= df0['Avg Session Length']
y = df0['Yearly Amount Spent']
plt.xlabel("Avg Session Length")
plt.ylabel("Yearly Amount Spent")
plt.scatter(x,y)
plt.show()


# In[42]:


x= df0['Time on App']
y = df0['Yearly Amount Spent']
plt.xlabel("Time on App")
plt.ylabel("Yearly Amount Spent")
plt.scatter(x,y)
plt.show()


# In[41]:


x= df0['Time on Website']
y = df0['Yearly Amount Spent']
plt.xlabel("Time on Website")
plt.ylabel("Yearly Amount Spent")

plt.scatter(x,y)
plt.show()


# In[40]:


x= df0["Length of Membership"]
y = df0['Yearly Amount Spent']
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")

plt.scatter(x,y)
plt.show()





