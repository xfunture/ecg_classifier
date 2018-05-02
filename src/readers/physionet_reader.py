from src.readers.base_reader import BaseReader
import pandas as pd
import wfdb
import keras


class PhysionetReader(BaseReader):
    def read_ecg_file(self, signal_file_path):
        return self.__read_signal_from_file(signal_file_path)

    def read_ecg_data(self, records_filename, labels_filename, signals_directory):
        signals = self.__read_signals_from_file(records_filename, signals_directory)
        labels = self.__read_labels_from_file(labels_filename)
        return signals, labels

    def __read_signals_from_file(self, records_filename, signals_directory):
        signals = []
        records_file = open(records_filename, 'r')
        for record_name in records_file:
            signal_file_path = '{}/{}'.format(signals_directory, record_name.strip())
            digital_signal = self.__read_signal_from_file(signal_file_path)
            signals.append(digital_signal)
        return signals

    def __read_signal_from_file(self, signal_file_path):
        record = wfdb.rdsamp(signal_file_path)
        digital_signal = record.adc()[:, 0]
        return digital_signal

    def __read_labels_from_file(self, labels_filename):
        labels_data = pd.read_csv(labels_filename, header=None, names=['id', 'label'])
        labels = labels_data['label']
        labels[labels == 'N'] = 0
        labels[labels == 'A'] = 1
        labels[labels == 'O'] = 2
        labels[labels == '~'] = 3
        labels = keras.utils.to_categorical(labels)  # Move to prepare data or sth?
        return labels


physionet_reader = PhysionetReader()
