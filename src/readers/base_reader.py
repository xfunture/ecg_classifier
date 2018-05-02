class BaseReader:
    def read_ecg_file(self, file_path):
        raise Exception('Method is not implemented')

    def read_ecg_data(self, records_filename, labels_filename, signals_directory):
        raise Exception('Method is not implemented')
