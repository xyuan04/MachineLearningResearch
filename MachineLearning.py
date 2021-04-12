# Import the Data
# Clean the Data
# Split the Data into Training/Test Sets
# Create a Model
# Train the Model
# Make Predictions
# Evaluate and Improve

# Popular libraries - Numpy, Pandas, MatPlotLib, Scikit-Learn
# Decision-Tree algorithm from library Scikit-Learn

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

hobby_data = pd.read_csv('MachineLearningExample.csv')
# print(hobby_data)
X = hobby_data.drop(columns=['hobby'])
y = hobby_data['hobby']
#training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
#print out the accuracy of the predictions 0 - 1
#print(score)

#tree.export_graphviz(model, out_file='hobby-predictor.dot', feature_names=['age', 'gender'], class_names=sorted(y.unique()), label= 'all', rounded=True, filled=True)

# predict the hobby of a 20 year old male, 29 year old female, 26 year old male and 31 year old female

#print(predict)
