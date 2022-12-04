#!/usr/bin/env python
# coding: utf-8

# <h1> Part 1 : Predict the price of a house <h1>
# <h2> Problem statement: The goal is to understand the relationship between house features and how these variables affect the house price.<h2>
#     

# In[1]:


# importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_excel("DS - Assignment Part 1 data set.xlsx")


# In[3]:


data.head()


# <h3>Checking data and Feature Engineering<h3>

# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


data.shape


# In[7]:


# Finding correlation 
data.corr()


# In[8]:


data['Transaction date']=data['Transaction date'].astype('str')


# In[9]:


data.dtypes


# In[10]:


# Extracting the year of transaction from the 'Transaction date' column

data['transaction_year']=data['Transaction date'].str.split('.').str[0]


# In[11]:


data.head()


# In[12]:


data=data.drop(['Transaction date'],axis=1)


# In[13]:


data.head()


# In[14]:


data['transaction_year']=data['transaction_year'].astype('float64')


# In[15]:


data.dtypes


# In[16]:


titles=list(data.columns)
titles


# In[17]:


# changing the positions of columns transaction_year and House price of unit area
titles[7],titles[8]=titles[8],titles[7]
titles


# In[18]:


data=data[titles]
data.head()


# <h3>Exploratory Data Analysis<h3>

# In[19]:


# Finding correlation between different features using Visualization technique

sns.heatmap(data.corr(),cmap='coolwarm')
plt.title('house_prices.corr()')


# <h4>From the plot it is clearly visible that the features are mostly independent of each other but Number of bedrooms and House size are somewhat correlated followed by Number of convenience stores near the house location, latitude and longitude<h4>

# In[20]:


# Plotting Average house age

sns.displot(data['House Age'],kde=True,bins=50)


# In[21]:


# Counting the Number of convenience stores

sns.countplot(x='Number of convenience stores',data=data)


# In[22]:


sns.jointplot(x='latitude',y='House price of unit area',data=data,kind='hex')


# In[23]:


sns.lmplot(x='latitude',y='House price of unit area',data=data,palette='coolwarm')


# In[24]:


sns.lmplot(x='longitude',y='House price of unit area',data=data,palette='coolwarm')


# <h4>We can clearly see that house prices are dependent on the location i.e the latitude and longitude of the house<h4>

# In[25]:


data['transaction_year'].value_counts()


# In[26]:


sns.jointplot(x='House Age',y='House price of unit area',data=data,kind='hex')


# <h4>No clear relationship between house age and house price from above plot<h4>

# In[27]:


sns.pairplot(data)


# <h3>Standardizing the variables using Standardscaler and then Training the data<h3>

# In[28]:


data.head()


# In[29]:


data.tail()


# In[30]:


X=data.drop('House price of unit area',axis=1)
y=data['House price of unit area']


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[33]:


from sklearn.preprocessing import StandardScaler


# In[34]:


scaler = StandardScaler()


# In[35]:


scaler.fit(data.drop('House price of unit area',axis=1))


# In[36]:


scaled_features = scaler.transform(data.drop('House price of unit area',axis=1))


# In[37]:


df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()


# <h3>Creating, Training and predicting the Model using Linear Regression<h3>

# In[38]:


from sklearn.linear_model import LinearRegression


# In[39]:


lr=LinearRegression()


# In[40]:


lr.fit(X_train,y_train)


# In[41]:


predictions = lr.predict(X_test)


# In[42]:


# Plotting chart between predictions and y_test

plt.scatter(y_test,predictions)


# In[44]:


lr.score(X_test,y_test)


# In[45]:


# Regression Evaluation metrics
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[46]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(),X, y ,cv = 5)
print(np.round(scores.mean(),4))


# <h3>Creating and Training the Model using Lasso Regression<h3>

# In[47]:


from sklearn import linear_model


# In[48]:


lasso_reg=linear_model.Lasso(alpha=50,max_iter=100,tol=0.5)


# In[49]:


lasso_reg.fit(X_train,y_train)


# In[50]:


lasso_reg.score(X_test,y_test)


# In[51]:


from sklearn.linear_model import LassoCV
scores = LassoCV(cv=5, random_state=0).fit(X, y)
scores.score(X, y)


# <h3>Creating and Training the Model using Ridge Regression<h3>

# In[52]:


from sklearn import linear_model


# In[53]:


ridge_reg=linear_model.Ridge(alpha=50,max_iter=100,tol=0.5)


# In[54]:


ridge_reg.fit(X_train,y_train)


# In[55]:


ridge_reg.score(X_test,y_test)


# In[56]:


from sklearn.linear_model import RidgeCV
scores = RidgeCV(cv=5).fit(X, y)
scores.score(X, y)


# <h3>Creating ,training and predicting the Model using Decision Tree and Random Forest Regressor<h3>

# In[58]:


# there is no need for scaling down the features here


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


X = data.drop('House price of unit area',axis=1)
y = data['House price of unit area']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[61]:


# Using Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor 


# In[62]:


dtree=DecisionTreeRegressor()


# In[63]:


dtree.fit(X_train,y_train)


# In[64]:


predictions = dtree.predict(X_test)


# In[65]:


dtree.score(X_test,y_test)


# In[66]:


scores = cross_val_score(DecisionTreeRegressor(),X, y ,cv = 5)
print(np.round(scores.mean(),4))


# In[67]:


# Hyperparameter Tuning 

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
param_grid = {
    'max_depth':[2,4,8,10,None],
    'criterion':['mse','mae'],
    'max_features':[0.25,0.5,1.0],
    'min_samples_split':[0.25,0.5,1.0]
}


# In[68]:


dec_reg = GridSearchCV(DecisionTreeRegressor(),param_grid=param_grid)


# In[69]:


dec_reg.fit(X_train,y_train)


# In[70]:


dec_reg.best_score_


# In[71]:


dec_reg.best_params_


# In[72]:


# Using Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor


# In[73]:


rfc = RandomForestRegressor(n_estimators=600)


# In[74]:


rfc.fit(X_train,y_train)


# In[75]:


predictions = rfc.predict(X_test)


# In[76]:


rfc.score(X_test,y_test)


# In[77]:


scores = cross_val_score(RandomForestRegressor(),X, y ,cv = 5)
print(np.round(scores.mean(),4))


# In[78]:


# predict the first row of the dataset
rfc.predict([[32.0,84.87882,10,24.98298,121.54024,1,575,2012.0]])


# In[79]:


plt.scatter(y_test,predictions)


# In[80]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
param_grid = {
    'max_depth':[2,4,8,10,None],
    'criterion':['mse','mae'],
    'max_features':[0.25,0.5,1.0],
    'min_samples_split':[0.25,0.5,1.0]
}


# In[81]:


random_reg = GridSearchCV(RandomForestRegressor(),param_grid=param_grid)


# In[82]:


random_reg.fit(X_train,y_train)


# In[83]:


random_reg.best_score_


# In[84]:


dec_reg.best_params_


# <h3>Creating and Training the Model using Ensemble technique (Bagging)<h3> 

# In[85]:


X=data.drop('House price of unit area',axis=1)
y=data['House price of unit area']


# In[86]:


from sklearn.model_selection import train_test_split


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[88]:


from sklearn.preprocessing import StandardScaler


# In[89]:


scaler = StandardScaler()


# In[90]:


scaler.fit(data.drop('House price of unit area',axis=1))


# In[91]:


scaled_features = scaler.transform(data.drop('House price of unit area',axis=1))


# In[92]:


df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()


# In[93]:


X_train.shape


# In[94]:


X_test.shape


# In[95]:


# Bagging technique using DecisionTreeRegressor

from sklearn.ensemble import BaggingRegressor
bag_model_dec=BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                           n_estimators=100,
                           max_samples=0.8,
                           oob_score=True,
                           random_state=0)
bag_model_dec.fit(X_train,y_train)
bag_model_dec.oob_score_


# In[96]:


bag_model_dec.score(X_test,y_test)


# In[97]:


# Bagging technique using RandomForestRegressor

from sklearn.ensemble import BaggingRegressor
bag_model_reg=BaggingRegressor(base_estimator=RandomForestRegressor(),
                            n_estimators=100,
                           max_samples=0.8,
                           oob_score=True,
                           random_state=0)
bag_model_reg.fit(X_train,y_train)
bag_model_reg.oob_score_


# In[98]:


bag_model_reg.score(X_test,y_test)


# <h3>Creating and Training the Model using Ensemble technique (Boosting)<h3>

# In[99]:


# Using GradientBoosting
from sklearn.ensemble import GradientBoostingRegressor
grad_decent = GradientBoostingRegressor()
scores = cross_val_score(grad_decent,X, y ,cv = 5)
print(np.round(scores.mean(),4))


# In[100]:


# Using XGboost
from xgboost import XGBRegressor
xbg = XGBRegressor()
scores = cross_val_score(xbg,X, y ,cv = 5)
print(np.round(scores.mean(),4))


# In[101]:


# Using Catboost
from catboost import CatBoostRegressor
catr = CatBoostRegressor()
scores = cross_val_score(catr,X, y ,cv = 5)
print(np.round(scores.mean(),4))


# In[102]:


# Using Lightboost
from lightgbm import LGBMRegressor
lgb = LGBMRegressor()
scores = cross_val_score(lgb,X, y ,cv = 5)
print(np.round(scores.mean(),4))


# <h3>Creating and Training the Model using Artificial Neural Network(ANN)<h3>

# In[127]:


X=data.drop('House price of unit area',axis=1)
y=data['House price of unit area']


# In[128]:


from sklearn.model_selection import train_test_split


# In[129]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[130]:


from sklearn.preprocessing import MinMaxScaler


# In[131]:


scaler=MinMaxScaler()


# In[132]:


scaler.fit(X_train)


# In[133]:


X_train=scaler.fit_transform(X_train)


# In[134]:


X_test=scaler.fit_transform(X_test)


# In[135]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[136]:


X_train.shape


# In[137]:


X_test.shape


# In[138]:


model=Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')


# In[139]:


model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),
          batch_size=128,epochs=400)


# In[140]:


losses=pd.DataFrame(model.history.history)


# In[141]:


losses.plot()


# <h4> From the plot we can see that the training loss and validation loss decreasing and being almost stable after that, giving us perfect behaviour and no overfitting <h4>

# In[142]:


model.evaluate(X_test, y_test, verbose=0)


# In[143]:


model.evaluate(X_train, y_train, verbose=0)


# In[144]:


test_pred=model.predict(X_test)


# In[145]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score


# In[146]:


np.sqrt(mean_squared_error(y_test, test_pred))


# In[147]:


mean_absolute_error(y_test, test_pred)


# <h3>Comparing the accuracies of all the models<h3>

# 1)Linear Regression                -   accuracy score -  64.30%,          cross_val_score  -  57.60% <h3> 

# 2)Lasso Regression                 -   accuracy score -  51.75%,          cross_val_score  -  48.68%  <h3> 

# 3)Ridge Regression                 -   accuracy score-   60.92%,           cross_val_score  -  56.66%<h3>

# 4)Decision Tree                    -accuracy score - 53.40%,            cross_val_score - 22.72%,
# score after hypertuning parameter  -  60.78%<h3>

# 4)Decision Tree                    -accuracy score - 74.61%,            cross_val_score - 67.49%,
# score after hypertuning parameter  -  61.69%<h3>

# 6)Bagging Techniques-          *) DecisionTreeRegressor- accuracy score - 75.49%, *) RandomForestRegressor-accuracy score-77.37%<h3> 

# 7)Boosting techniques- *) Gradient Descent - accuracy score-68.47%,
#   *) XGBoost - accuracy score- 64.25%,                            
#   *) CatBoost- accuracy score-71.66% ,                 
#   *)LightBoost - accuracy score-69.11%<h3>

# <h3>Drawbacks of each technique's Assumptions<h3>

# 1) Linear Regression - As we know that linear regression assumes that the data is independent of each other, which is not true especially in our case. The latitude and longitude of the house location were dependent with each other in predicting the house price. Also linear regression is sensitive to outliers. That is why I am not recommending this algorithm model.<h4> 

# 2) Lasso Regression - We can see that the accuracy score and cross validation score of Lasso regression is low compared to other model scores. In Lasso regression, when there are correlated variables, it retains only one variable and sets other correlated variables to zero. That will possibly lead to some loss of information resulting in lower accuracy in our model. Thus I cannot recommend this algorithm for our model.<h4>

# 3) Ridge Regression - The main issue with Ridge regression is that it follows same assumptions as linear regression. Thus, I cannot recommend this algorithm for our model due to low accuracy.<h4>

# 4) Decision Tree - It randomly selects node randomly to be its root which automatically results in lower accuracy than any other models. Also, as the number of splits increases in a decision tree, the time required to build the tree also increases. Trees with a large number of splits are however prone to overfitting resulting in poor accuracy. This is the reason I do not recommend this algorithm for this model.<h4>

# 5) Random Forest - Our model has performed considerably well when using random forest algorithm. The reason is that Random forest adds additional randomness to the model while growing trees. When splitting a node, it searches for the best feature among a random subset of features instead of looking for the most important feature. Thus, it reduces the overfitting problem in decision trees and lessens the variance, improving accuracy. I would recommend this algorithm for this model.<h4>

# 6) Bagging Techniques - As we know that both bagging and random forests are ensemble-based algorithms that aim to reduce the complexity of models that overfit the training data. But the main aim of bagging technique is to train a bunch of unpruned decision trees on different random subsets of the training set, sampling with replacement. While in our case, both algorithms have great accuracy score, bagging is effective by reducing the complexity of overfitting models. We would too recommend this algorithm<h4> 
# 
# 

# 7) Boosting Techniques - Boosting is a resilient method that curbs over-fitting easily. One disadvantage of boosting is that it is sensitive to outliers since every classifier is obliged to fix the errors in the predecessors.It can be considered but as validation score is less, I cannot recommend this algorithm <h4> 

# 8) Artificial Neural Network(ANN) - We are just comparing the other algorithms of machine learning with the ANN here. There is not much improvement in the ANN algorithm so we would not be recommending this too.<h4>

# <h3>Final Recommendation - Based on the accuracy score and the type of dataset, I would recommend the random forest algorithm or bagging technique. It is because both are great while handling outliers and doesn't allow model to overfit.<h3>

# In[ ]:




