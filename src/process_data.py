import sys
import os
import keras
from src.utils.preprocessor import process_signals
from src.readers.physionet_reader import physionet_reader
from src.utils.persistance import save_processed_data
import src.constants as constants


if __name__ == '__main__':
    records_filename = os.path.join(constants.PROJECT_PATH, sys.argv[1])
    labels_filename = os.path.join(constants.PROJECT_PATH, sys.argv[2])
    signals_directory = os.path.join(constants.PROJECT_PATH, sys.argv[3])

    signals, labels = physionet_reader.read_ecg_data(records_filename, labels_filename, signals_directory)
    labels = keras.utils.to_categorical(labels)
    spectrograms, labels = process_signals(signals, labels)
    save_processed_data(constants.ECG_DATA_PATH, (spectrograms, labels))
