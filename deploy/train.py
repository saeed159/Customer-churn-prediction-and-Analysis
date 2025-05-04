from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import mlflow
import json

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


from imblearn.over_sampling import SMOTE #for Data augmentation
from model_utils import ClfSwitcher, format_params, save_model
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score # Import metrics

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
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and then transform it
    # X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test data using the fitted scaler
    # X_test_scaled = scaler.transform(X_test)
    #Data Augmentation because off the noted imbalance
    # sm = SMOTE(random_state = 2)
    # X_train_scaled, Y_train = sm.fit_resample(X_train_scaled, Y_train)
    

    # steps = [('clf', ClfSwitcher())]  # Scaler is applied separately
    
    steps = [('scaler', StandardScaler()), ('clf', ClfSwitcher())]


    
    pipeline = Pipeline(steps)
    parameters = [
        format_params(LogisticRegression(), C=[0.1, 0.5, 1.0]),
        format_params(DecisionTreeClassifier(random_state=42), max_depth=[3, 5, None], min_samples_leaf=[1, 3]),
        format_params(LinearDiscriminantAnalysis()),
        format_params(RandomForestClassifier(random_state=42), n_estimators=[50, 100], max_depth=[5, None], min_samples_leaf=[2, 5]),
        format_params(SVC(random_state=42), kernel=['linear', 'rbf'], C=[0.1, 1.0]),
        format_params(KNeighborsClassifier(), n_neighbors=[3, 5, 7]),
        
                     
        format_params(GaussianNB())
    ]
    gscv = GridSearchCV(pipeline, parameters, cv=3, scoring='accuracy')
    # mlflow start logging
    with mlflow.start_run():
        print('training model')
        gscv.fit(X_train, Y_train)  # Train on the scaled training data

        print('all tested models:')
        for i, params in enumerate(gscv.cv_results_['params']):
            model = gscv.estimator.set_params(**params)
            model.fit(X_train, Y_train)  # Train each model on scaled training data
            y_pred = model.predict(X_test)  # Predict on test dat

            run_name = f"model_{i}"
            with mlflow.start_run(nested=True, run_name=run_name):
                print(f"  - Logging model {i} with parameters: {params}")
                mlflow.log_params(params)
                mlflow.sklearn.log_model(model, f"model_{i}")

                mean_accuracy = gscv.cv_results_['mean_test_score'][i]
                mlflow.log_metric("mean_test_accuracy", mean_accuracy)

                precision = precision_score(Y_test, y_pred)
                recall = recall_score(Y_test, y_pred)
                f1 = f1_score(Y_test, y_pred)
                conf_matrix = confusion_matrix(Y_test, y_pred)
                class_report = classification_report(Y_test, y_pred, output_dict=True)

                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_text(json.dumps(conf_matrix.tolist()), "confusion_matrix.json")
                mlflow.log_text(json.dumps(class_report), "classification_report.json")

        # Log the best model separately with all metrics
        print('\nbest model: \n\n', gscv.best_estimator_)
        mlflow.sklearn.log_model(gscv.best_estimator_, "best_model")
        mlflow.log_params(gscv.best_estimator_.get_params())

        best_y_pred = gscv.predict(X_test)  # Predict with the best model on scaled test data
        best_accuracy = gscv.score(X_test, Y_test)  # Evaluate on scaled test data
        best_precision = precision_score(Y_test, best_y_pred)
        best_recall = recall_score(Y_test, best_y_pred)
        best_f1 = f1_score(Y_test, best_y_pred)
        best_conf_matrix = confusion_matrix(Y_test, best_y_pred)
        best_class_report = classification_report(Y_test, best_y_pred, output_dict=True)

        mlflow.log_metric("best_model_accuracy", best_accuracy)
        mlflow.log_metric("best_model_precision", best_precision)
        mlflow.log_metric("best_model_recall", best_recall)
        mlflow.log_metric("best_model_f1_score", best_f1)
        mlflow.log_text(json.dumps(best_conf_matrix.tolist()), "best_model_confusion_matrix.json")
        mlflow.log_text(json.dumps(best_class_report), "best_model_classification_report.json")

        save_model(gscv)
    # 
    # print('training model')
    # gscv.fit(X_train,Y_train)
    # # print results
    # print('best model: \n\n', gscv.best_estimator_)
    # print('\ntest set accuracy: ', gscv.score(X_test, Y_test))
    # save_model(gscv)

if __name__ == 'main':
    main()
