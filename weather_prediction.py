# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the csv file and making dataframe object 
input_data=pd.read_csv("C:\\Users\\DELL\\Desktop\\Weather_pred\\weather-dataset\\weatherHistory.csv",skiprows=0)

# Dropping Unnecessary features
input_data = input_data.drop(["Humidity", "Loud Cover", "Pressure (millibars)", "Wind Bearing (degrees)"], axis = 1)

# Separating the dependent_variables(y) and independent_variables(X)
X = input_data.iloc[:, 3:-2].values
y = input_data.iloc[:, 1].values


# It will give 
# count of data in summary column,
# count of unique class
# top class(the class that is occuring most)
# frequency of top class 
input_data["Summary"].describe()

# It will give frequency of each classes present in summary column
input_data['Summary'].value_counts()

# plotting counter plot different classes
plt.figure()
sns.countplot(x="Summary", data=input_data)
plt.show()


# Manual labeling based on their occurence i.e in 6 classes: 'Partly Cloudy', 'Mostly Cloudy', 'OverCast',
# 'Clear', 'Foggy', 'other' 
for i in range(len(y)):
    if y[i]=="Partly Cloudy":
        y[i]=0
    elif y[i]=="Mostly Cloudy":
        y[i]=1
    elif y[i]=="Overcast":
        y[i]=2
    elif y[i] == "Clear":
        y[i] = 3
    elif y[i] == "Foggy":
        y[i] = 4
    else:
        y[i] = 5

# changing data-type from object to int
y=y.astype('int32')


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# applying DecisionTreeClassifier model
from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier(random_state=0)
clf1.fit(X_train, y_train)

# predicting based on model clf1
y_pred = clf1.predict(X_test)

# applying LogisticRegression model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", max_iter = 100)
clf.fit(X_train, y_train)

# predicting based on model clf
y_pred = clf.predict(X_test)

# checking the accuracy of the model
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
print(accuracy_score(y_test, y_pred))
print(r2_score(y_test, y_pred))
