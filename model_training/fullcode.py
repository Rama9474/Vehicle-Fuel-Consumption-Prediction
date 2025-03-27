#importing required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
# from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import warnings
warnings.filterwarnings('ignore')


#read the dataset

df = pd.read_excel(r'C:\Users\komma\Documents\fuel-consumption-prediction\data\measurements2.xlsx')
print(df.head())


#check null values

import seaborn as sns
sns.heatmap(df.isnull())
df.isnull()
null_values=df.isnull().sum().sort_values(ascending=False)
ax=sns.barplot(x=null_values.index,y=null_values)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
import matplotlib.pyplot as plt
plt.show()
#removing null values
df.drop(['refill gas','refill liters','specials'],axis=1,inplace=True)
sns.heatmap(df.isnull())

#handling null values
temp_inside_mean=np.mean(df['temp_inside'])

print(temp_inside_mean)

df['temp_inside'].fillna(temp_inside_mean,inplace=True)
sns.heatmap(df.isnull())

#seperating independent and dependent variables

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x=df.drop(['consume','gas_type'],axis=1)
y=df['consume']

x.columns

x=x.values
y=y.values

#Splitting Data Into Train And Test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Training The Model In Multiple Algorithms

#Linear Regression
linReg = LinearRegression()
linReg.fit(x_train,y_train)

x_train.shape

y_pred = linReg.predict(x_test)
print(y_pred)
print(linReg.coef_,linReg.intercept_)
accuracy = linReg.score(x_test,y_test)
print(accuracy)
dum1 = pd.get_dummies(df['gas_type'])
print(dum1)
df=pd.concat([df,dum1],axis=1)
df.drop(['gas_type'],axis=1,inplace=True)
x1=df.drop(['consume'],axis=1)
y1=df['consume']
x1.columns
x1=x1.values
y1=y1.values
x_train.shape
x_train[0]

#Lasso Regression model

lassoReg = linear_model.Lasso(alpha = 0.1)
lassoReg.fit(x,y)
y_pred = lassoReg.predict(x_test)
print(y_pred)
accuracy = lassoReg.score(x_test,y_test)
print(accuracy)

#SVM MODEL

svr = SVR().fit(x,y)
y_pred = svr.predict(x_test)
accuracy = svr.score(x_test,y_test)
print(accuracy)

#Decision Tree Model:

dt = DecisionTreeRegressor(random_state = 0)
dt.fit(x,y)
y_pred = dt.predict(x_test)
print(y_pred)
accuracy = dt.score(x_test,y_test)
print(accuracy)

#Random Forest Model:

rf = RandomForestRegressor(n_estimators = 100 , random_state = 0)
rf.fit(x,y)
y_pred = rf.predict(x_test)
print(y_pred)

accuracy = rf.score(x_test,y_test)
print(accuracy)


#Testing Model With Multiple Evaluation Metrics
#Compare the Model

# Assuming 'x_test' is available in the environment and is a pandas DataFrame or a NumPy array.
y_pred = linReg.predict(x_test)  # Predict on the entire x_test dataset

print("Prediction Evaluation using Linear Regression")
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R-squared:', r2_score(y_test, y_pred))


y_pred = lassoReg.predict(x_test)
print("Prediction Evaluation using lasso Regression")
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R-squared:', r2_score(y_test, y_pred))


y_pred = svr.predict(x_test)
print("Prediction Evaluation using svr Regression")
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R-squared:', r2_score(y_test, y_pred))

y_pred = dt.predict(x_test)
print("Prediction Evaluation using decisiontree Regression")
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R-squared:', r2_score(y_test, y_pred))


y_pred = rf.predict(x_test)
print("Prediction Evaluation using Random Regression")
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R-squared:', r2_score(y_test, y_pred))

# import pickle
# pickle.dump(dt,open('fuel2.pkl','wb'))

# from flask import Flask, render_template, request
# import pickle

import os
import pickle

# Define the correct path to the 'model' directory in the app folder
model_dir = os.path.join(os.path.dirname(os.getcwd()), 'app', 'model')

# Create the 'model' directory if it doesn't already exist
os.makedirs(model_dir, exist_ok=True)

# Choose the model you want to save (e.g., 'dt' for DecisionTreeRegressor)
model = dt  # Replace 'dt' with any model you've trained, like 'rf' or 'linReg'

# Define the full path to the model file in the 'model' directory
model_path = os.path.join(model_dir, 'fuel2.pkl')

# Save the trained model using pickle
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model saved successfully at {model_path}")

