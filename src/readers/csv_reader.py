from src.readers.base_reader import BaseReader
import pandas as pd


class CSVReader(BaseReader):
    def read_signals_from_file(self, records_filename, signals_directory):
        signals = []
        records_file = open(records_filename, 'r')
        for record_name in records_file:
            signal_file_path = '{}/{}.csv'.format(signals_directory, record_name.strip())
            digital_signal = self.read_signal_from_file(signal_file_path)
            signals.append(digital_signal)
        return signals

    def read_signal_from_file(self, signal_file_path):
        record = pd.read_csv(signal_file_path, header=None, names=['signal'])
        digital_signal = record['signal'].as_matrix()
        return digital_signal


csv_reader = CSVReader()
