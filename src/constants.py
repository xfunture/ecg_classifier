import os

PROJECT_PATH = os.path.abspath('.')
ECG_DATA_PATH = os.path.join(PROJECT_PATH, 'data/ecg_data.pkl')
MODEL_PATH = os.path.join(PROJECT_PATH, 'models/cnn_model.h5')
TEST_RESULTS_PATH = os.path.join(PROJECT_PATH, 'results/cnn_results.txt')
