"""## Importing the libraries"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

#Load the data
df = pd.read_csv('./data/kidney_disease.csv')
df.head()

#Get the shape to the data (the no. of rows and column)
df.shape

#Create columns names to keep 
columns_to_retain = ['sg','al','sc','hemo','pcv','rc','wc','htn','classification']

#Drop the columns that are not in columns_to_retain
df = df.drop([col for col in df.columns if not col in columns_to_retain ],axis=1)
#Drop the rows with na or missing values
df = df.dropna(axis=0)

#Transform the non-numeric data in the columns 
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column] = LabelEncoder().fit_transform(df[column])

#Split the data into independent (X) and dependent (Y)
X = df.drop(['classification'],axis=1)
y = df['classification']

#Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

#Build the Model
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
pickle.dump(classifier, open('./models/ChronicDisease_model.pkl', 'wb'))

# Loading model to compare the results (read)
model = pickle.load(open('./models/ChronicDisease_model.pkl', 'rb'))
'''