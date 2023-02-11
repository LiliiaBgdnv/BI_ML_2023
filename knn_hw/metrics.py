import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    true_positive = y_pred[(y_true == y_pred) & (y_pred == 1)].shape[0]
    false_positive = y_pred[(y_true != y_pred) & (y_pred == 1)].shape[0]
    true_negative = y_pred[(y_true == y_pred) & (y_pred == 0)].shape[0]
    false_negative = y_pred[(y_true != y_pred) & (y_pred == 0)].shape[0]
       
    if true_positive + true_negative == 0:
        precision, recall, f1 = 0, 0, 0
    elif false_positive + false_negative == 0:
        precision, recall, f1 = 1, 1, 1
    else: 
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    accuracy = (true_positive + true_negative) / len(y_pred)
    return precision, recall, f1, accuracy




def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    
    return sum(y_true == y_pred) / len(y_pred)



def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))



def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    return np.sum((y_true - y_pred)**2) / y_pred.shape[0]



def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return np.sum(np.abs(y_true - y_pred)) / y_pred.shape[0]

    