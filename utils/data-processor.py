import numpy as np
from scipy.stats import iqr

def detect_outliers(data):
    """Detect outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr_val = iqr(data)
    
    lower_bound = q1 - 1.5 * iqr_val
    upper_bound = q3 + 1.5 * iqr_val
    
    return [x for x in data if lower_bound <= x <= upper_bound]

def calculate_moving_averages(data, window_sizes=[5, 10, 20]):
    """Calculate multiple moving averages"""
    return {
        f'ma_{w}': np.convolve(data, np.ones(w)/w, mode='valid')
        for w in window_sizes
    }

def calculate_momentum(data, period=5):
    """Calculate momentum indicator"""
    return np.diff(data, n=period)

def k_fold_cross_validation(data, k=5):
    """Implement k-fold cross-validation"""
    fold_size = len(data) // k
    return [
        (data[i*fold_size:(i+1)*fold_size], data[:i*fold_size] + data[(i+1)*fold_size:])
        for i in range(k)
    ]