# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load dataset: Load the Iris dataset using load_iris().

2.Create DataFrame: Convert the dataset into a pandas DataFrame.

3.Split features and target: Separate features (X) and target (y) from the DataFrame.

4.Split data into training and testing sets: Use train_test_split() to split the data into training and test sets.

5.Initialize SGDClassifier: Instantiate an SGDClassifier() with appropriate parameters.

6.Train the model: Fit the model on the training data using sgd_clf.fit().

7.Make predictions: Predict the target values on the test data using sgd_clf.predict().

8.Calculate accuracy: Evaluate the model's accuracy using accuracy_score().

9.Confusion matrix: Compute the confusion matrix using confusion_matrix().

10.Generate classification report: Generate a classification report with classification_report().
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Divya R
RegisterNumber:  212222040040
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())
X=df.drop('target',axis=1)
y=df['target']
X_train,X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cn=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cn)
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)



```

## Output:

## df.head()
![image](https://github.com/user-attachments/assets/e66f3b8c-0abe-4bdd-8549-2d2f3b119d7d)
## Accuracy
![image](https://github.com/user-attachments/assets/31b38a8a-0fac-4039-bbdf-e510519924a0)
## Confusion matrix
![image](https://github.com/user-attachments/assets/064daf9c-1a9f-4f93-a1e2-07892d6e339e)
## Classification report
![image](https://github.com/user-attachments/assets/fbee3c30-051d-444d-99a9-dbb03f46fc24)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
