import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, classification_report , cohen_kappa_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay


def get_model_name(model):
    return type(model).__name__

def get_metrics(model, y_true, X_true, predictions=None):
    y_pred = model.predict(X_true)

    if isinstance(model, SVC):
        # Use decision_function for SVC to compute decision scores
        decision_scores = model.decision_function(X_true)
        y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
    else:
        y_pred_proba = model.predict_proba(X_true)[:, -1]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificty = tn / (tn+fp)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificty,
        'F1-Score': f1,
        'Kappa': kappa,
        'Balanced Accuracy': (sensitivity +specificty )/2,
        'False Positive Ratio': fpr,
        'True Positive Ratio': tpr,
        'Thresholds': thresholds,
        'Area Under the Curve': auc,
        "Gini": round(2 * auc - 1, 2),
        'Confusion Matrix': cm,
        'Classification Report': report
    }

    return metrics

def get_metrics_table(model, y_true, X_true, model_name=None, predictions=None):
    y_pred = model.predict(X_true)

    if isinstance(model, SVC):
        decision_scores = model.decision_function(X_true)
        y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
    else:
        y_pred_proba = model.predict_proba(X_true)[:, -1]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificty = tn / (tn+fp)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    kappa = cohen_kappa_score(y_true, y_pred)


    if model_name is None:
        model_name = get_model_name(model)

    metrics = {
        "Model": model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificty,
        'Balanced Accuracy': (sensitivity +specificty )/2,
        'Kappa': kappa,
        'F1-Score': f1,
        'Area Under the Curve': auc,
        "Gini": round(2 * auc - 1, 2)
    }

    df = pd.DataFrame([metrics])

    return df

def display_metrics(train_metrics_report, test_metrics_report, not_show=['Confusion Matrix', 'Classification Report', 'False Positive Ratio', 'True Positive Ratio', 'Thresholds']):
    for k in train_metrics_report.keys():
        if k not in not_show:
            print(k, " - Train : ", round(train_metrics_report[k], 4))
            print(k, " - Test : ", round(test_metrics_report[k], 4))
            print("-" * 100)

def plot_classification_metrics(model, y_true, X_true, save_path='img/classification_metrics/', model_name=None):
    if model_name is None:
        model_name = get_model_name(model)

    y_pred = model.predict(X_true)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    RocCurveDisplay.from_estimator(model, X_true, y_true).plot(ax=ax2)

    ax1.set_title('Confusion Matrix ' + model_name)
    ax2.plot([0, 1], [0, 1], 'k--', label='Benchmark')
    ax2.set_title('ROC Curve Prediction ' + model_name)
    ax2.legend()

    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot(ax=ax1)
    
    
    save_path = save_path + model_name + ".pdf"

    
    
    # Save the entire figure as a PDF
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path, bbox_inches='tight', format='pdf')

    # Close the figure after saving
    plt.close()
    fig.savefig(save_path, format='pdf')
    return fig


