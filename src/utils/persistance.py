import pickle
from sklearn.metrics import confusion_matrix


def save_processed_data(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_processed_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def save_results(file_path, model_name, epochs, y_actual, y_predict):
    with open(file_path, 'w') as file:
        file.write('Model: {}\nEpochs: {}\nConfusion Matrix:\n{}'
                   .format(model_name, epochs, confusion_matrix(y_actual, y_predict)))
