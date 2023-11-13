import pandas as pd
import numpy as np


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,confusion_matrix ,precision_score, recall_score, f1_score, classification_report ,roc_curve, roc_auc_score ,roc_curve, auc, ConfusionMatrixDisplay , RocCurveDisplay


from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import utils_FRED_MD.model_evaluation  as me
SPLIT = 0.7
SEED=42
mms = MinMaxScaler()
scaler = StandardScaler()



def evaluate_models(X, y, models_and_params=None, scoring='accuracy', cross_val = StratifiedKFold(n_splits=10, shuffle=False),preprocessing=MinMaxScaler(),split=SPLIT, seed=SEED,print_metrics=False):
    """
    Evaluate multiple models with optional grid search for hyperparameter tuning.

    Parameters:
    - X: Features
    - y: Target variable
    - models_and_params: List of tuples where each tuple contains a model and its corresponding grid parameters (default is None)
    - scoring: Scoring metric for cross-validation (default is 'accuracy')
    - cv: Number of folds for cross-validation (default is 10)
    - seed: Random seed for reproducibility (default is 7)

    Returns:
    - results: List of dictionaries containing model names, mean scores, and standard deviations
    """



    if models_and_params is None:
        models_and_params = [
            ('LR', LogisticRegression(class_weight='balanced'), {'C': [0.001, 0.01, 0.1, 1, 5,10,8,30,50, 100, 1000],'penalty': ['elasticnet', 'l1', 'l2'],"solver":['saga',"lblinear"]}),
            ('LDA', LinearDiscriminantAnalysis(), {}),
            ('KNN', KNeighborsClassifier(), {'n_neighbors': [2,4,10,int(np.sqrt(len(y)))]}),
            ('CART', DecisionTreeClassifier(), {}),
            ('NB', GaussianNB(), {}),
            ('SVM', SVC(), {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'kernel':['linear','rbf']}),
            ("RandomForest",RandomForestClassifier(),{})
        ]

    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed, stratify=y)
    if preprocessing is None == False:

        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.transform(X_test)

    
    list_results = []

    for name, model, params in models_and_params:
        

        
        grid_search = GridSearchCV(model, params,cv=cross_val, scoring=scoring,n_jobs=-1)
        grid_result = grid_search.fit(X_train, y_train)

        print(f"Best parameters for {name}: {grid_result.best_params_}")
        print(f"Best Train {scoring} for {name}: {grid_result.best_score_}")

        best_estimator = grid_result.best_estimator_

        train_metrics = me.get_metrics(best_estimator ,y_train,X_train )
        test_metrics = me.get_metrics(best_estimator ,y_test,X_test )
        if print_metrics:
            me.display_metrics(train_metrics,test_metrics)


        me.plot_classification_metrics(best_estimator ,y_train,X_train ,model_name= name + "_train")
        me.plot_classification_metrics(best_estimator ,y_test,X_test,model_name= name + "_tests" )

        table_train = me.get_metrics_table(best_estimator , y_train, X_train)
        table_train['Dataset_partition'] =  'Train'
        table_test = me.get_metrics_table(best_estimator , y_test, X_test)
        table_test['Dataset_partition'] =  'Test'

        model_results = {
            "Name": name,
            'Metrics_Train':table_train,
            'Metrics_Test':table_test,
            "Best_Estimator": best_estimator,
            "Best Paramns": grid_result.best_params_
        }
        list_results.append(list_results)    
    return list_results


