Heart Diseases Prediction
#IMPORTING THE DEPENDENCIES

import pandas as pd    #pandas is useful for creating dataframe, dataframe is nothing but a structured table, data in csv file, so csv file represents comma separated values and so pandas put it in a more structured form.
import numpy as np #use to make some numpy arrays like list
from sklearn.model_selection import train_test_split  #we need to split original data into train and test data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #evaluate our model, to check how well our model is performing

Data collection and Processing

#loading csv data to a pandas dataframe
heart_df=pd.read_csv(r"C:\Users\Sahil\Downloads\heart.csv")

#print first 5 rows of the dataset
heart_df.head()

#No. of rows and columns in the dataset
heart_df.shape

#Getting some info about data
heart_df.info()

#statistical measures about the data
heart_df.describe()

#Checking for missing values
heart_df.isnull().sum()

heart_df.nunique()

#Checking the distribution of Target variable 
heart_df['target'].value_counts()

X=heart_df.drop(columns='target',axis=1) #while dropping a column put axis =1 and in case of row put axis=0
Y=heart_df['target']
print(X)

print(Y) #Target data

Splitting the Data into Training and Testing Data
We create 4 variables here, X_train,X_test,Y_train,Y_test .

Splitting this x into x train and x test. It means the x train features are separated as features of all the training data and x test contain the features of all the test data.

And Y train contains the target of all these features present in this x train and the y test contain the corresponding target for x test.

We will put some parameters in split(), size = 0.2 means how much percentage of the data you want as test data, so point represent 20% of the data and we use stratify=Y, it means it will split the data, like x test contain similar proportion of 1 & 0 as present in the original dataset if we do not use stratify = y, there is a posiblity that all the values in the x test may contain 0 or 1. We put random_state =2, so it is just to split the data in a specific way

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=2)

#Lets check the number of training and testing data we have)
print(X.shape,X_train.shape,X_test.shape)

LOGISTIC REGRESSION

model=LogisticRegression()

#Training the LogisticRegression model with Training Data
#.fit will try to find the relationship or pattern between the features and Target.

model.fit(X_train, Y_train)

ACCURACY SCORE

#Accuracy on Training Data

X_train_prediction = model.predict(X_train) #helps to predict the target value
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print('Accuracy on Training Data :',training_data_accuracy) 

#Accuracy on Test Data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

#To get the input values (Input values i.e features). We will create list or tuple as input data.
input_data = [43, 0, 0, 132, 341, 1, 0, 136, 1, 3, 1, 0, 3]


#Change the input data (tuple) to a numpy array because numpy array is easy to reshape instead of tuple.
input_data_as_numpy_arrays = np.asarray(input_data)

#Reshape the numpy array as we are predicting for only one instance, means we are telling ML model that we are predicting only for 
#one data point not all 242 values.
input_data_reshaped = input_data_as_numpy_arrays.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

print(prediction)

if (prediction[0] == 0):
    print("The Person doesn't have heart diseases")
else:
    print("The person has heart diseases")


