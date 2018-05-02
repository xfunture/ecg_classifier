# ecg_classifier

This is an automatic electrocardiogram (ECG) detector using convolutional neural networks. 

There are various types of cardiac arrhymias. Among them, atrial fibrillation (AF) is most common.  The disease is associated with significant mortality and morbidity through an increasing risk of heart failure, dementia, and stoke. Detection for AF remains problematic, because the signal may be episodic and often episodes have no symptoms. 

This dectector focuses on identifying AF from other kinds of records, namely normal sinus rhythm, other abnormal rhythms and noisy recordings. 

ECG recordings are usually stored as 1-D waveform, which displays changes in amplitude of electical activies of the heart over time. Through Fourier transformation, the waveform are turned into spectrogram - a 2-D visual representation of the spectrum of frequencies with amplitude represented by brightness.

A deep convolutional network, which is specialized for processing an image, were trained to classify the spectrograms. 

I used the [keras](https://keras.io/) package with Tensorflow as backend to train the model on AWS p2.xlarge instance with NVIDIA Tesla® K80 GPUs.  


### Dataset
Total 8331 single short ECG recordings with 30s length were collected (thanks to AliveCor). These recordings were labeled in four classes: normal(59%) , AF(9%), other(30%), and noise(2%). 

<img src="https://github.com/gogowenzhang/ECG_Detector/blob/master/img/ecg_new.png" width='600' height='500'>

### Data Processing
Transform 1-D waveform into 2-D spetrogram by Fourier transformation. 
Log transformation and standardization were applied to spectrograms before passed into model. 

<img src="https://github.com/gogowenzhang/ECG_Detector/blob/master/img/data_processing.png" width='600' height='450'>

### Model Architecture

Convolutional layers are arranged in blocks. For each block there are four convolutional layers, following each convolutional layer, there is one normalization layer, one relu activation layer and one dropout layer. A max pooling layer is added at the end of each block. 

Following the convolutional layers, a customized layer is added to take average of features across time. Then there is a flatten layer to reduce dimension before passing to classifer(fully-connected) layer. 

A standard linear layer with Softmax is used to compute the class probabilities. 

<img src="https://github.com/gogowenzhang/ECG_Detector/blob/master/img/nn.png" width="350" height="500">

### Load ECG data
Download ECG data [here](https://physionet.org/challenge/2017/training2017.zip) and extract training2017 folder to data directory.  

### How to Run
#### Install python requirements
```
pip install --requirement requirements.txt
``` 

#### Process data
```
python src/process_data.py RECORDS_FILENAME LABELS_FILENAME SIGNALS_DIRECTORY
```
Pickle file with processed data will be stored in `data/ecg_data.pkl`.

#### Train and save model
```
python src/train_cnn.py EPOCHS
```
CNN model will be stored in model `models/cnn_model.h5`. Testing results will be stored in `results/cnn_results.txt`.

##### Predict
```
python src/predict.py SIGNAL_FILE
```
Probability of each class will be printed on screen (Normal, Atrial Fibrillation, Other, Noise).

### Example

```bash
python src/process_data.py data/training2017/RECORDS data/training2017/REFERENCE.csv data/training2017
python src/train_cnn.py 20
python src/predict.py data/training2017/A00003
```





