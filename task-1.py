#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve


# In[2]:


df = pd.read_csv("After_Modifications.csv")
df.head(10)


# In[4]:


y = df['price']
X = df.drop(columns = ['price'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[5]:


linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)
r_squared = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.title("Residual Plot")
plt.show()
print("R-squared:", r_squared)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)


# # Feature Selection (RFE)

# In[6]:


num_features = 5
rfe = RFE(linear, n_features_to_select = num_features)
rfe.fit(X_train, y_train)
selected_features = X.columns[rfe.support_]


# In[7]:


selected_features


# In[8]:


y = df['price']
X_selected = X[selected_features]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_selected, y, test_size = 0.2, random_state = 42)


# In[9]:


linear2 = LinearRegression()
linear2.fit(X_train2, y_train2)
y_pred2 = linear2.predict(X_test2)
r_squared2 = r2_score(y_test2, y_pred2)
mae2 = mean_absolute_error(y_test2, y_pred2)
mse2 = mean_squared_error(y_test2, y_pred2)
rmse2 = np.sqrt(mse2)
plt.scatter(y_test2, y_pred2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()
residuals2 = y_test2 - y_pred2
plt.scatter(y_pred2, residuals2)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.title("Residual Plot")
plt.show()
print("R-squared:", r_squared2)
print("Mean Absolute Error:", mae2)
print("Mean Squared Error:", mse2)
print("Root Mean Squared Error:", rmse2)


# # Lasso Regression

# In[10]:


desired_alpha = 0.1
model = Lasso(alpha = desired_alpha)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
selected_features = coefficients[coefficients['Coefficient'] != 0]
print("Selected Features:")
print(selected_features)


# # Target Feature Analysis

# In[11]:


correlation_matrix = df.corr()
target_correlation = correlation_matrix['price']
relevant_features = target_correlation[(target_correlation >= 0.5) | (target_correlation <= -0.5)].index.tolist()


# In[12]:


relevant_features


# In[13]:


plt.figure(figsize = (12, 10))
sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm")
plt.title("Correlation Matrix")
plt.show()


# In[14]:


y = df['price']
X_selected2 = X[['sqft_living', 'grade', 'sqft_above', 'sqft_living15']]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_selected2, y, test_size = 0.2, random_state = 42)


# In[15]:


linear3 = LinearRegression()
linear3.fit(X_train3, y_train3)
y_pred3 = linear3.predict(X_test3)
r_squared3 = r2_score(y_test3, y_pred3)
mae3 = mean_absolute_error(y_test3, y_pred3)
mse3 = mean_squared_error(y_test3, y_pred3)
rmse3 = np.sqrt(mse3)
plt.scatter(y_test3, y_pred3)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()
residuals3 = y_test3 - y_pred3
plt.scatter(y_pred3, residuals3)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.title("Residual Plot")
plt.show()
print("R-squared:", r_squared3)
print("Mean Absolute Error:", mae3)
print("Mean Squared Error:", mse3)
print("Root Mean Squared Error:", rmse3)


# # Cross Validation

# In[16]:


k = KFold(n_splits = 10, shuffle = True, random_state = 42)
accuracy = cross_val_score(linear, X, y, cv = k) 
print("The accuracy using cross validation is: ", accuracy)
average_accuracy = np.mean(accuracy)
print(f'Average Accuracy: {average_accuracy:.2f}')


# In[17]:


train_sizes, train_scores, test_scores = learning_curve(linear, X, y, cv = k, train_sizes = np.linspace(0.1, 1.0, 10)).
avg_train = np.mean(train_scores, axis = 1)
avg_test = np.mean(test_scores, axis = 1)
for i, train_size in enumerate(train_sizes): # Since it prints each training and testing score for each fold (each training size)
    print(f"Train Size: {train_size:.1f}, Average Training Score: {avg_train[i]:.2f}, Average Testing Score: {avg_test[i]:.2f}")


# In[18]:


plt.figure(figsize = (10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis = 1), 'o-', label = 'Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis = 1), 'o-', label = 'Testing Score')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy Score')
plt.title('Learning Curves (Linear Regression)')
plt.legend(loc = 'best')
plt.grid(True)
plt.show()

