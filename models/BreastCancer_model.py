"""## Importing the libraries"""
import numpy as np
import pandas as pd
import pickle

"""## Importing the dataset"""
dataset = pd.read_csv('./data/breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values  # Independent variable
y = dataset.iloc[:, -1].values  # Dependent variable

"""## Splitting the dataset into the Training set and Test set"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""## Training the Simple Linear Regression model on the Training set"""
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""
y_pred = classifier.predict(X_test)

"""## Accuracy"""
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))

'''
# Saving model to disk
pickle.dump(classifier, open('./models/BreastCancer_model.pkl', 'wb'))

# Loading model to compare the results (read)
model = pickle.load(open('./models/BreastCancer_model.pkl', 'rb'))
'''