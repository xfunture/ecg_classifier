import pickle
from sklearn.metrics import confusion_matrix
import os

def save_processed_data(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_processed_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def save_results(file_path, model_name, epochs, y_actual, y_predict):
    path = os.path.dirname(file_path)
    if os.path.exists(path) == False:
        os.mkdir(path)
    with open(file_path, 'w') as file:
        file.write('Model: {}\nEpochs: {}\nConfusion Matrix:\n{}'
                   .format(model_name, epochs, confusion_matrix(y_actual, y_predict)))
