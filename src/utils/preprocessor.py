import numpy as np
import scipy


def process_signals(signals, labels):
    trimmed_signals, new_labels = trim_signals(signals, labels)
    spectrograms = fourier_transform(trimmed_signals)
    return spectrograms, new_labels


def trim_signals(signals, labels):
    trimmed_signals = []
    related_labels = []
    for i in range(len(signals)):
        signal = signals[i]
        label = labels[i]
        if len(signal) >= 18000:
            first_part = signal[:9000]
            second_part = signal[9000:18000]
            trimmed_signals.extend((first_part, second_part))
            related_labels.extend((label, label))
        elif len(signal) >= 9000:
            trimmed_signals.append(signal[:9000])
            related_labels.append(label)
    return np.array(trimmed_signals), np.array(related_labels)


def process_single_signal(signal):
    trimmed_signal = trim_single_signal(signal)
    spectrograms = fourier_transform(np.array([trimmed_signal]))
    return spectrograms


def trim_single_signal(signal):
    if len(signal) < 9000:
        raise Exception('Signal is too short. Pick another one')
    return signal[:9000]


def fourier_transform(signals):
    """
    Transform 1-D array signals into 2-D array as spectrogram.
    Take log and standardize spectrogram.
    Input: 1-D array signals, labels
    Output: 2-D array spectrograms
    """
    spectrogram = list()
    for signal in signals:
        spectrogram.append(signal_spectrogram(signal))
    spectrogram = np.array(spectrogram)

    # Log transformation and standardization
    log_spectrogram = np.log(spectrogram + 1)
    centers = log_spectrogram.mean(axis=(1, 2))
    standard_deviation = log_spectrogram.std(axis=(1, 2))
    spectrograms_standardized = \
        np.array([(x - c) / d for x, c, d in zip(log_spectrogram, centers, standard_deviation)])
    spectrograms_extended = spectrograms_standardized[..., np.newaxis]
    return spectrograms_extended


# Concert signal to spectrogram
def signal_spectrogram(signal):
    _, _, Sxx = scipy.signal.spectrogram(signal,
                                         fs=300,
                                         window=('tukey', 0.25),
                                         nperseg=64,
                                         noverlap=0.5,
                                         return_onesided=True)

    return Sxx.T
