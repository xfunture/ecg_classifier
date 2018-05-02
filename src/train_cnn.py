import sys
from sklearn.cross_validation import train_test_split
from src.models.cnn import train, test
import src.constants as constants
from src.utils.persistance import load_processed_data, save_results

if __name__ == '__main__':
    epochs = int(sys.argv[1])

    x, y = load_processed_data(constants.ECG_DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = train(x_train, x_test, y_train, y_test, epochs=epochs, model_path=constants.MODEL_PATH)

    print('Testing...')

    y_predict, y_actual = test(model, x_test, y_test)
    save_results(constants.TEST_RESULTS_PATH, 'CNN', epochs, y_actual, y_predict)
