import os
import sys
import pdb
import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC

def train_model():

    basepath = ""

    # Reading the data from my system 
    data = []
    with open(basepath + 'train.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)

            
    X = []  
    Y = []

    # Excluding the data from ROW 1 which is our Column Name Row. Seperating our Input data from our Output data.
    for data_row in data[1:]:
        Z = lambda x: [255 if int(y)>0 else 0 for y in data_row[1:]]
        X.append(Z(data_row))
        Y.append([int(data_row[0])])

    X = np.array(X)
    Y = np.array(Y)

    # Training set from the same distribution
    train_X = X[1:40000]
    train_Y = Y[1:40000]

    # Testing set from the same distribution
    test_X = X[40000:]
    test_Y = Y[40000:]

    # Create a model
    # clf = DTC()
    clf = RFC(n_estimators=50) # number of trees i want in this forest 
    

    print("Training Models...")

    # Fitting the model [Training the model using our training dataset train_X & train_Y]
    clf.fit(train_X,train_Y)

    # Store the model on our system at a specified path
    joblib.dump(clf,basepath + 'model_rfc.pkl')

    # Predict on test data from same distribution
    pred_Y = clf.predict(test_X)
    print("Accuracy of trained model on MNIST TEST DATASET is: ",accuracy_score(test_Y,pred_Y) * 100,"%")
    print("Confusion_matrix: ",confusion_matrix(test_Y,pred_Y))



def predict_value(test_X):
    basepath = ""

    # Load the already created model
    clf = joblib.load(basepath + './Model/model_rfc.pkl')

    # Predict on test data
    pred_Y = clf.predict(np.array(test_X))

    return pred_Y

if __name__ == '__main__':
    train_model()
    