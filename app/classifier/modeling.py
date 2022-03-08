from typing import Tuple
import pandas as pd
import pickle
import numpy as np
import os

from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier



def read_csv(path:str) -> pd.DataFrame():
    data = pd.read_csv(path)
    return data.rename(columns= {'HeartDiseaseorAttack':'target'})

def preprocess_data(data:pd.DataFrame()) -> Tuple[list,list,list,list]: 
    return train_test_split(data.drop(['target'], axis = 1), data.target, test_size= 0.3, random_state= 42)

def train_classifier(**dingsbums) -> any:
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    classifier = dingsbums['classifier'].fit(dingsbums['x_train'],dingsbums['y_train'])
    return classifier

def predict_classifier(classifier: RandomForestClassifier, x_test: list) -> float:
    return classifier.predict(x_test)

def calculate_metrics(y_prediction: float, y_test: list) -> Tuple[float,float]:
    accuracy = accuracy_score(y_test, y_prediction)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prediction)
    auc_score = roc_auc_score(y_test,y_prediction)
    return accuracy, auc_score

def save_model(file_name:str, model):
    path = os.getcwd()+"\\app\\classifier\\model\\"+file_name
    pickle.dump(model, open(path, 'wb'))


if __name__ == '__main__':
    currentClassifier = RandomForestClassifier()
    df = read_csv(path=os.getcwd()+'\\app\classifier\data\heart_disease_health_indicators_BRFSS2015.csv')
    x_train, x_test, y_train, y_test = preprocess_data(data=df)
    classifier = train_classifier(x_train=x_train, y_train=y_train, classifier=currentClassifier)
    y_prediction = predict_classifier(classifier=classifier, x_test=x_test)
    accuracy, auc_score = calculate_metrics(y_prediction=y_prediction, y_test=y_test)
    filename="rf_model_"+str(date.today())+".sav"
    save_model(file_name = filename, model=classifier)
    print(accuracy)
    print(auc_score)