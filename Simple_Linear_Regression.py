"""
Created on Mon Sep 25 00:32:40 2017
@author: mahmoud
"""


import numpy as np  #mathematical tools"""
import matplotlib.pyplot as plt  #sub library of matplotlib, help us plot nice charts"""
import pandas as pd 

#-- Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')    #2 columns YearsExperience and Salary

                                            # Years Experienc
X = dataset.iloc[:, :-1].values             # [ Lines , Columns ] of data set ':' means all , ':-1' means all columns except last one

                                            # Salary
y = dataset.iloc[:, 1].values               # 1 is the number of the column with Years Experienc being no. 0


#-- Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

                                             #test size is 20% of X and y
                                             #random_state we will have random results if we dont put it
                                             #we have 30 example we will make 20 as training and 10 as testing set 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

"""                    
#-- Feature Scaling
                    #we are scaling the age and salary becase they are not in the same level
                    #and sometimes we will need to do it for decreasing the run time like in decission tress it will run for a long time if we didn't do it
                    #we don't need to apply feature scaling to y for now bec. they are only 0 and 1

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()                  #sc for scale, object of the class
X_train = sc_X.fit_transform(X_train)    #when applying scalling to your training set you have to fit the scalling set and then transform it
X_test = sc_X.transform(X_test)          #dont need to fit it bec. its alreay fit in train set
                        #All the Varibales will be from -1 to +1
"""

# Fitting simple Linear Regression to the training set, make the machine learn the relation between yearExperience and salary
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()              #create object of the class
regressor.fit(X_train, y_train)             #fit the regressor item into the training set


# Predicting the Test set results
y_pred = regressor.predict(X_test)           #the vector of predictions of the dependant variable


# Visualising the training set results
plt.scatter(X_train, y_train, color = 'red')    #make points in graph, x coordinate then y then choose color
plt.plot(X_train, regressor.predict(X_train), color = 'blue')      #make line, y coordi = see how he could have predicted our red points, predict how he predict our already made red points
plt.title('Salary vs Experience (Training set)')    #just improve its UI
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()                                      #end of graph and ready to plot it



# Visualising the test set results
plt.scatter(X_test, y_test, color = 'red')    #make points in graph, x coordinate then y then choose color
plt.plot(X_train, regressor.predict(X_train), color = 'blue')     #we should not change it with test because we made it by the train set(20 example) #make line, y coordi = see if we predicted our test set as how it should have been?
plt.title('Salary vs Experience (Test set)')    #just improve its UI
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()                                      #end of graph and ready to plot it











#y coordi = will be the something we are trying to guess


