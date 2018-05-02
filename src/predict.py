import sys
import os
from keras.models import load_model
from src.readers.physionet_reader import physionet_reader
from src.utils.preprocessor import process_single_signal
from src.models.cnn import predict
from src.utils.metrics import k_f1_score
import src.constants as constants

if __name__ == '__main__':
    signal_file_path = os.path.join(constants.PROJECT_PATH, sys.argv[1])

    model = load_model(constants.MODEL_PATH, custom_objects={'k_f1_score': k_f1_score})
    signal = physionet_reader.read_ecg_file(signal_file_path)
    spectrograms = process_single_signal(signal)
    prediction_probability = predict(model, spectrograms)
    print("Prediction: {}".format(prediction_probability[0]))
