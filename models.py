from flask import Flask
import pandas as pd

from sklearn.ensemble import RandomForestClassifier



def Random_forest_model(X_train, y_train):
    model = RandomForestClassifier(max_depth=10,
                                       random_state=2,
                                       n_estimators=139)
    model.fit(X_train, y_train)
    return model

# function for moel predictions and accuracies
def getPrediction(inputs, model):
    input_pred = model.predict(inputs)
    return input_pred

def getAccuracy(model, X_test,y_test):
    accuracy = round(model.score(X_test,y_test)*100, 2)
    return accuracy
