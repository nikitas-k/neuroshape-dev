import numpy as np

def normalize_data(data):
    data_normalized = np.subtract(data, np.mean(data, axis=0))
    data_normalized = np.divide(data_normalized, np.std(data_normalized, axis=0))
    
    return data_normalized