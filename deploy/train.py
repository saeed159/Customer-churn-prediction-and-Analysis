from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

# from src.utils.data_utils import load_data, stratified_split
# from src.utils.model_utils import ClfSwitcher, format_params, save_model

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # for scaling
from sklearn.model_selection import train_test_split # for splitting data set into training and validation
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix #metrices
from imblearn.over_sampling import SMOTE #for Data augmentation
from model_utils import ClfSwitcher, format_params, save_model
from sklearn.linear_model import SGDRegressor

colors = ["green", "blue", "red", "yellow", "orange", "purple", "pink", "black",  "gray", "cyan", "magenta"]
# loading the dataset

# clear

def main():
    data_s=pd.read_csv("Churn_for_bank_customers.csv")
    data=data_s.drop(columns=['Row Nu','Customer Id','Surname'], axis=1, inplace=False)
    # encoding the gender column
    G_encode = LabelEncoder()
    labels = G_encode.fit_transform(data['Gender'])
    data['gender'] = labels
    #encoding Geography because of the three values
    Geo_labeles=pd.get_dummies(data["Geography"],prefix="Geography").astype(float)
    data = pd.concat([data, Geo_labeles], axis=1)
    #drop outliers
    Outliers_col=["Credit score","Age","NumOfProducts"]
    for f in Outliers_col:
        q1=data[f].quantile(0.25)
        q3=data[f].quantile(0.75)
        lower_bnd=q1-1.5*(q3-q1)
        upper_bnd=q3+1.5*(q3-q1)
        data=data[(data[f]<= upper_bnd) & (data[f]>=lower_bnd)]
    data = data.drop(columns=["Geography"])
    data.drop(columns='Gender', axis=1, inplace=True)

    # Splitting the dataset features into X and y
    X = data.drop(columns=["Exited"]) #Drop Traget column
    Y = data["Exited"] # only target column
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    steps = [('scaler', StandardScaler()), ('clf', ClfSwitcher())]

    sgd_regressor = SGDRegressor(
        max_iter=1000,
        tol=1e-3,
        learning_rate="adaptive",# Adaptive learning rate
        eta0=0.1,  # Initial learning rate
        random_state=42
        )
    
    pipeline = Pipeline(steps)
    parameters = [format_params(LogisticRegression(),
                                C=[0.5, 0.75, 1]),
                  format_params(sgd_regressor),#for Adaptive learnin g
                  format_params(DecisionTreeClassifier(),
                                max_depth=[2, 3, None],
                                min_samples_leaf=[1, 2]),
                  format_params(LinearDiscriminantAnalysis())
                  ]
    gscv = GridSearchCV(pipeline, parameters, cv=3, scoring='accuracy')
    print('training model')
    gscv.fit(X_train,Y_train)
    # print results
    print('best model: \n\n', gscv.best_estimator_)
    print('\ntest set accuracy: ', gscv.score(X_test, Y_test))
    save_model(gscv)

if __name__ == 'main':
    main()
