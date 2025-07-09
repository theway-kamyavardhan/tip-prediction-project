
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.preprocessing import StandardScaler , LabelEncoder

#loading the dataset
df = pd.read_csv("tips.csv")
print(df)

#converting the categorical data into numerical value using Label encoder
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_day = LabelEncoder()
le_time = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['day'] = le_day.fit_transform(df['day'])
df['time'] = le_time.fit_transform(df['time'])

#Lets define the independent and dependent variable 
X = df[['total_bill','sex','smoker','day','time','size']]
y = df[['tip']]

#now using StandardScaler we will convert the scale of the X variable
scale = StandardScaler()
X_scaled = scale.fit_transform(X)

#lets split the data now
X_train , X_test ,Y_train , Y_test = train_test_split(X_scaled,y,random_state=42,test_size=0.3)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#lets start our linear regression model
model = LinearRegression()
model.fit(X_train , Y_train)

#Now create a variable to store the predicted values
y_pred = model.predict(X_test)
print(y_pred)

#now validate the results by r2 score and mean squared error
r2 = r2_score(Y_test,y_pred)
mse = mean_squared_error(Y_test,y_pred)
print(f"Model R² Score: {round(r2, 3)}")
print(f"Mean Squared Error: {round(mse, 3)}")

#lets plot now 
plt.scatter(Y_test,y_pred,alpha=0.4)
plt.plot([Y_test.min()[0], Y_test.max()[0]], [Y_test.min()[0], Y_test.max()[0]], color='red')
plt.title("Actual Tip vs Predicted tip")
plt.xlabel("Actual tip")
plt.ylabel("Predicted tip")
plt.grid()
plt.show()