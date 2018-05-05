class BaseReader:
    def read_ecg_file(self, signal_file_path):
        return self.read_signal_from_file(signal_file_path)

    def read_ecg_data(self, records_filename, labels_filename, signals_directory):
        signals = self.read_signals_from_file(records_filename, signals_directory)
        labels = self.read_labels_from_file(labels_filename)
        return signals, labels

    def read_signals_from_file(self, records_filename, signals_directory):
        raise Exception('Method is not implemented')

    def read_signal_from_file(self, signal_file_path):
        raise Exception('Method is not implemented')

    def read_labels_from_file(self, labels_filename):
        raise Exception('Method is not implemented')
