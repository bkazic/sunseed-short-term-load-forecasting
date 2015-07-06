import numpy as np
import warnings

# MAPE - Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Use of this metric is not recommended; for illustration only. 
    See other regression metrics on sklearn docs:
      http://scikit-learn.org/stable/modules/classes.html#regression-metrics

    Use like any other metric
    >>> y_true = [3, -0.5, 2, 7]; y_pred = [2.5, -0.3, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    Out[]: 24.791666666666668
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if y_pred has any zero elements. If yes, remove them, and raise warning
    zero_indices = np.flatnonzero(y_true == 0)
    if (zero_indices.size != 0):
        y_true = np.delete(y_true, zero_indices)
        y_pred = np.delete(y_pred, zero_indices)
        
        warning_msg = "Found {0} zero elements in y_pred. Removing {0} zero elements".format(len(zero_indices))
        warnings.warn(warning_msg, RuntimeWarning)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import numpy as np


# MASE - Mean Absolute Scaled Error
def mean_absolute_scaled_error(training_series, naive_training_series, testing_series, prediction_series):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forecast error for univariate time series prediction.

    See `"Another look at measures of forecast accuracy" <http://robjhyndman.com/papers/another-look-at-measures-of-forecast-accuracy/>`_, Rob J Hyndman


    :param list   training_series: the series used to train the model
    :param list   testing_series: the test series to predict
    :param list   prediction_series: the prediction of testing_series (same size as testing_series)
    """
    training_series = np.array(training_series)
    testing_series = np.array(testing_series)
    prediction_series = np.array(prediction_series)
    n = training_series.shape[0]
    #d = np.abs(np.diff(training_series)).sum() / (n - 1)
    d = np.abs(training_series - naive_training_series).sum() / (n - 1)

    errors = np.abs(testing_series - prediction_series)
    return errors.mean() / d
