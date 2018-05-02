import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras import backend as K
from src.utils.metrics import k_f1_score


GENERATOR_BATCH_SIZE = 32
# In general should be len(x_test) / GENERATOR_BATCH_SIZE or x_test.shape[0] / GENERATOR_BATCH_SIZE
GENERATOR_TEST_BATCHES = 10


def train(x_train, x_test, y_train, y_test, epochs=10, model_path=None):
    train_generator = generator(x_train, y_train)
    test_generator = generator(x_test, y_test)
    model = build_model(train_generator, test_generator, epochs, x_train[0].shape)
    if model_path:
        model.save(model_path)
    return model


def test(model, x_test, y_test):
    test_generator = generator(x_test, y_test)
    y_predict = model.predict_generator(test_generator, GENERATOR_TEST_BATCHES).argmax(axis=1)
    y_actual = y_test[:GENERATOR_TEST_BATCHES * GENERATOR_BATCH_SIZE].argmax(axis=1)
    return y_predict, y_actual


def predict(model, x):
    return model.predict(x)


def generator(x, y):
    """
    Initialize data generator for model training
    Input: training data and testing data
    Output: image generator
    """
    return ImageDataGenerator().flow(x, y, batch_size=GENERATOR_BATCH_SIZE)


def build_model(train_generator, test_generator, epochs, shape):
    """
    load training data and testing data, compile and train CNN model, return training history
    Parameters
    Input: train_generator, test_generator
    epochs: number of epochs for training
    Output: training history parameters
    """
    model = Sequential()

    # Convolutional layer
    model = add_conv_blocks(model, 4, 6, initial_input_shape=shape)

    # Feature aggregation across time
    model.add(Lambda(lambda x: K.mean(x, axis=1)))

    model.add(Flatten())

    # Linear classifier
    model.add(Dense(4, activation=K.softmax))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy', k_f1_score])

    model.fit_generator(train_generator,
                        validation_data=test_generator,
                        steps_per_epoch=shape[0] / GENERATOR_BATCH_SIZE,
                        epochs=epochs,
                        verbose=1,
                        validation_steps=5)
    return model


# Convolutional layers
def add_conv_blocks(model, block_size, block_count, initial_input_shape):
    channels = 32
    for i in range(block_count):
        for j in range(block_size):
            if (i, j) == (0, 0):
                conv = Conv2D(channels, kernel_size=(5, 5),
                              input_shape=initial_input_shape, padding='same')
            else:
                conv = Conv2D(channels, kernel_size=(5, 5), padding='same')
            model.add(conv)
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.15))
            if j == block_size - 2:
                channels += 32
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.15))
    return model
