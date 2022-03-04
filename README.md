# Heart-Diseases-Prediction
The model should predict their target, weather the person has defective or not. 
IMPORTING THE DEPENDENCIES

import pandas as pd #pandas is useful for creating dataframe, dataframe is nothing but a structured table, data in csv file,
#so csv file represents comma separated values and so pandas put it in a more structured form.
import numpy as np #use to make some numpy arrays like list
from sklearn.model_selection import train_test_split #we need to split original data into train and test data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #evaluate our model, to check how well our model is performing
​
Data collection and Processing
#loading csv data to a pandas dataframe
heart_df=pd.read_csv(r"C:\Users\Sahil\Downloads\heart.csv")
#print first 5 rows of the dataset
heart_df.head()
age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
0	63	1	3	145	233	1	0	150	0	2.3	0	0	1	1
1	37	1	2	130	250	0	1	187	0	3.5	0	0	2	1
2	41	0	1	130	204	0	0	172	0	1.4	2	0	2	1
3	56	1	1	120	236	0	1	178	0	0.8	2	0	2	1
4	57	0	0	120	354	0	1	163	1	0.6	2	0	2	1
Target represents 1 means persons have heart diseases and 0 represenys person doesn't have heart diseases.

#No. of rows and columns in the dataset
heart_df.shape
(303, 14)
#Getting some info about data
heart_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trestbps  303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalach   303 non-null    int64  
 8   exang     303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slope     303 non-null    int64  
 11  ca        303 non-null    int64  
 12  thal      303 non-null    int64  
 13  target    303 non-null    int64  
dtypes: float64(1), int64(13)
memory usage: 33.3 KB
#statistical measures about the data
heart_df.describe()
age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
count	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000
mean	54.366337	0.683168	0.966997	131.623762	246.264026	0.148515	0.528053	149.646865	0.326733	1.039604	1.399340	0.729373	2.313531	0.544554
std	9.082101	0.466011	1.032052	17.538143	51.830751	0.356198	0.525860	22.905161	0.469794	1.161075	0.616226	1.022606	0.612277	0.498835
min	29.000000	0.000000	0.000000	94.000000	126.000000	0.000000	0.000000	71.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
25%	47.500000	0.000000	0.000000	120.000000	211.000000	0.000000	0.000000	133.500000	0.000000	0.000000	1.000000	0.000000	2.000000	0.000000
50%	55.000000	1.000000	1.000000	130.000000	240.000000	0.000000	1.000000	153.000000	0.000000	0.800000	1.000000	0.000000	2.000000	1.000000
75%	61.000000	1.000000	2.000000	140.000000	274.500000	0.000000	1.000000	166.000000	1.000000	1.600000	2.000000	1.000000	3.000000	1.000000
max	77.000000	1.000000	3.000000	200.000000	564.000000	1.000000	2.000000	202.000000	1.000000	6.200000	2.000000	4.000000	3.000000	1.000000
Percentile means there are 25% of the values are less than 47 age

#Checking for missing values
heart_df.isnull().sum()
age         0
sex         0
cp          0
trestbps    0
chol        0
fbs         0
restecg     0
thalach     0
exang       0
oldpeak     0
slope       0
ca          0
thal        0
target      0
dtype: int64
heart_df.nunique()
age          41
sex           2
cp            4
trestbps     49
chol        152
fbs           2
restecg       3
thalach      91
exang         2
oldpeak      40
slope         3
ca            5
thal          4
target        2
dtype: int64
#Checking the distribution of Target variable 
heart_df['target'].value_counts()
1    165
0    138
Name: target, dtype: int64
1 represents Defective Heart whereas,
0 represents Healthy Heart

165 people have heart diseases, whereas 138 doesn't have heart diseases. The distribution of have and doesn't have are quite similar. Importance of finding it is that, we have almost equal number of distribution in the two classes, so here the difference isn't very much. If 80% of the values lies in one class and 20% of the values lies in another class, than that's become a problem. So in that cases we need to do some processing.

Splitting the Features and Target
Except Target all the columns are act as features, so we remove target from the features data frames and store it separately. That's why we split the features and Target.

We create 2 variables x and y, and we store features column in variable x and target column in the variable y.

X=heart_df.drop(columns='target',axis=1) #while dropping a column put axis =1 and in case of row put axis=0
Y=heart_df['target']
print(X)
     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \
0     63    1   3       145   233    1        0      150      0      2.3   
1     37    1   2       130   250    0        1      187      0      3.5   
2     41    0   1       130   204    0        0      172      0      1.4   
3     56    1   1       120   236    0        1      178      0      0.8   
4     57    0   0       120   354    0        1      163      1      0.6   
..   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...   
298   57    0   0       140   241    0        1      123      1      0.2   
299   45    1   3       110   264    0        1      132      0      1.2   
300   68    1   0       144   193    1        1      141      0      3.4   
301   57    1   0       130   131    0        1      115      1      1.2   
302   57    0   1       130   236    0        0      174      0      0.0   

     slope  ca  thal  
0        0   0     1  
1        0   0     2  
2        2   0     2  
3        2   0     2  
4        2   0     2  
..     ...  ..   ...  
298      1   0     3  
299      1   0     3  
300      1   2     3  
301      1   1     3  
302      1   1     2  

[303 rows x 13 columns]
print(Y) #Target data
0      1
1      1
2      1
3      1
4      1
      ..
298    0
299    0
300    0
301    0
302    0
Name: target, Length: 303, dtype: int64
Splitting the Data into Training and Testing Data
We create 4 variables here, X_train,X_test,Y_train,Y_test .

Splitting this x into x train and x test. It means the x train features are separated as features of all the training data and x test contain the features of all the test data.

And Y train contains the target of all these features present in this x train and the y test contain the corresponding target for x test.

We will put some parameters in split(), size = 0.2 means how much percentage of the data you want as test data, so point represent 20% of the data and we use stratify=Y, it means it will split the data, like x test contain similar proportion of 1 & 0 as present in the original dataset if we do not use stratify = y, there is a posiblity that all the values in the x test may contain 0 or 1. We put random_state =2, so it is just to split the data in a specific way

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=2)
#Lets check the number of training and testing data we have)
print(X.shape,X_train.shape,X_test.shape)
(303, 13) (242, 13) (61, 13)
Model Training
LOGISTIC REGRESSION

model=LogisticRegression()
#Training the LogisticRegression model with Training Data
#.fit will try to find the relationship or pattern between the features and Target.
model.fit(X_train, Y_train)
C:\Users\Sahil\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
LogisticRegression()
Model Evaluation
ACCURACY SCORE

#Accuracy on Training Data
X_train_prediction = model.predict(X_train) #helps to predict the target value
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
We are going to use accuracy as our evaluation metric. It will be asked to predict the target and this predicted value compare with the original target values. If the model has predicted and about 99 values are correct, then in that case the accuracy score is 99%.

print('Accuracy on Training Data :',training_data_accuracy) 
Accuracy on Training Data : 0.8512396694214877
Accuracy more than 75% is somewhat good. And here we are getting an accuracy score of 85%. It means out of 100 predictions our model can vary correctly for 85 values. This we get because of less data (3,3), We have used only 242 values for training. We can get more than that by using large dataset.

#Accuracy on Test Data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Test Data :',test_data_accuracy)
Accuracy on Test Data : 0.819672131147541
As we can see the accuracy score of test data is almost 82% and for the training data, it is 85%. It means it's a good result. Beacause the accuracy score of training and testing data should be almost similar to each other. If the accuracy score on training is too large, and accuracy score on test data is too small, it means our model is overfitted.

Building a predictive system
In this system if we give the values, so all the feature values, such as, the age of the person,their sex and all these features, the model should predict their target, weather the person has defective or not. So we are going to do this in the predictive system.

#To get the input values (Input values i.e features). We will create list or tuple as input data.
input_data = [43, 0, 0, 132, 341, 1, 0, 136, 1, 3, 1, 0, 3]
​
​
#Change the input data (tuple) to a numpy array because numpy array is easy to reshape instead of tuple.
input_data_as_numpy_arrays = np.asarray(input_data)
​
#Reshape the numpy array as we are predicting for only one instance, means we are telling ML model that we are predicting only for 
#one data point not all 242 values.
input_data_reshaped = input_data_as_numpy_arrays.reshape(1,-1)
​
prediction = model.predict(input_data_reshaped)
​
print(prediction)
​
[0]
We can see here, our model is predicting 1. The target 1, which means person has heart defective.

if (prediction[0] == 0):
    print("The Person doesn't have heart diseases")
else:
    print("The person has heart diseases")
The Person doesn't have heart diseases
