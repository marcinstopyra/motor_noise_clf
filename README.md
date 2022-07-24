# Motor Noise Classifier

## Dataset description and problem statement

Dataset consists of 23 around one-minute long audio records of electric motors during work. The motors were labeled by a qualified worker as the 'good ones' which passed the quality assurance(QA), and as 'bad ones', which sound was either not pleasant for a user or indicating some defects. However, despite the training and the experience of the worker, this method of performing QA does not show sufficient repeatability. The labeling is highly dependent on the current condition of the worker. Therefore, to automate an automatic QA, a binary classifier was developed using a Convolutional Neural Network.

## Notebooks
The experiments and models were developed for three different sample lengths: 1,2 and 5 seconds. For clarity, the project will be described here with with use of the notebooks for 1 sec samples
1. [Data Exploration and Preprocessing](https://github.com/marcinstopyra/motor_noise_clf/blob/master/00_data_exploration_and_preprocess_1sec.ipynb)
2. [Feature Extraction and KMean Clustering](https://github.com/marcinstopyra/motor_noise_clf/blob/master/10_Cluster_1sec_samples.ipynb)
3. [Final Model Development](https://github.com/marcinstopyra/motor_noise_clf/blob/master/20_CNN_1sec_samples.ipynb)

## Data Exploration and Preprocessing
### Data Exploration
In the 1st notebook the data exploration was performed. The signals were visualised in different domains:
- Time domain: 
![example of silent motor signal in time domain](/images/time_domain1.png)
![example of loud motor signal in time domain](/images/time_domain2.png)

- frequency domain (after performing FFT on a signal):
![example of silent motor signal in frequency domain](/images/freq_domain1.png)
![example of loud motor signal in frequency domain](/images/freq_domain2.png)

Observation was made that the best way to differ good motors from bad will be by creating spectrograms from signals and feeding them to CNN.

### Data Preprocessing
The dataset consists of only 23 audio recordings, therefore the decision was made to segmentise them into short 1, 2 and 5 seconds long sections with given overlap. By performing this the number of samples raised up to the order of 1000.

For each segment a Melspectrogram was made. The reasoning behind the decision to use this type of spectrogram was that the data was labeled with respect to the personal impression of the worker trained to do it. Melspectrograms are scaled to show the best how we humans perceive sounds.

In the end the ready melspectrograms together with data about the motors were put into a form of pandas dataframe table and saved as csv files. Before this melspectrograms were flattened to make saving in this format possible. The original size of spectrograms together with parameters with which they were created (n_fft, hop_length) was saved in a json file.

## Feature Extraction and KMean Clustering
In 2nd notebook the problem of mislabeled data was handled as some of the motors (e.g.5, 11, 22, 23) seemed to be recognised wrongly. To cope with this problem decision was made to cluster the samples after feature extraction performed with use of stacked AutoEncoder

Optimal AutoEncoder hyperparameters were found by performing grid search for several parameters configurations. The AutoEncoder was trained with use of kFold validation method.

Feature extraction was then performed on all samples with use of an Encoder submodel. Out of those features the notable features for motor classification was performed by rejecting the features which were always the same, no matter of motor class. These features were found by checking ranges of all features. For some of them the range was 0 meaning that this feature holds no information on motor class.

On reduced dataset TSNE reduction was performed for better visualisation and then KMean clustering. The clustered data was then visualised on a heatmap showing how many samples from each motor were assigned to each cluster. Based on that the decision was made to change labels of several motors (on picture in blue circles)


![heatmap of clustered samples](/images/heatmap1.png)

## Final Model Development
### Dataset splitting
A custom way of data splitting was performed in order to split dataset with respect to motors. The reasoning behind this decision is that the samples originating from the same audio recording are really similar, so a standard split after shuffling would lead to really good accuracy results that could not necessarily turn out well in the real world, when the new motor audio would be presented to the model. The final train-test split in the notebook is:
- 13 motors in train set ($\approx 56$%)
- 10 motors in test set ($\approx 44$%)

### Model
Convolutional Neural Network hyperparameters were optimised by checking several parameters in order:
1. filter and layers numbers
2. batch size
3. activation function

Each training was performed with use of kFold validation. The training histories for the best model was then visualised, showing quick overfitting for some train-validation dataset splits.


![training histories](/images/training_history1.png)

This effect was quite expected as the dataset is really small. To cope with this problem the dropout layers were tested.

Final model was trained on the full train dataset and tested on test set resulting in 0.9685 accuracy.

The custom accuracy was then introduced due to the fact that the final prediction of motor class is a sum of predictions made on all of the segments of the given motor noise audio. 
An example of such prediction is shown in the table:


![results table](/images/pred_table.png)

Because of dataset small size the huge variation of results was expected depending on train-test split. Therefore, the final model was cross validated on several splits (15 random train-test splits). The results of cross validations:
- Minimal accuracy: 70%
- Maximal accuracy: 100%
- Mean accuracy: 96%
- Variance: 0.007733

The baseline accuracy for the model is 56%