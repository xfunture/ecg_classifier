from keras import backend as K


def k_f1_score(y_actual, y_predicted):
    """
    Customized F1 score for Keras model log
    Input: y_true, y_pred
    Output: f1 score
    """
    y_predicted = K.argmax(y_predicted, axis=1)
    y_actual = K.argmax(y_actual, axis=1)
    result = []
    for i in range(3):
        denom = (K.sum(K.cast(K.equal(y_actual, i), dtype='float32')) +
                 K.sum(K.cast(K.equal(y_predicted, i), dtype='float32'))) + K.epsilon()
        num = K.sum(K.cast(K.equal(y_actual, i), dtype='float32') * K.cast(K.equal(y_predicted, i), dtype='float32'))
        result.append(2.0 * num / denom)
    return (result[0] + result[1] + result[2]) / 3.0
