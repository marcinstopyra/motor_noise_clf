import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json
from sklearn.preprocessing import normalize
from random import randint
from itertools import product
from sklearn.cluster import KMeans
import seaborn as sns
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Reshape, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import layers
from tensorflow.compat.v1.keras.backend import get_session
import sys


##################################################################################
# Data Exploration and preporcessing

class Motor:
    """ an object containing data of a single motor from dataset
    ---
    attributes:
        id:     motor id number -> int
        isGood: motor label -> bool / int (0,1)
        signal: audio record of the motor during work -> np.array / list
        FFT:    signal transformed with Fast Fourier Transform
    ---
    methods:
        createSegments: function segmentizes a signal into segments of desired length 
        
    """
    def __init__(self, isGood, signal, index):
        self.id = index
        self.isGood = isGood 
        self.signal = signal
        self.FFT = None

    def createSegments(self, segment_length, shift=None, samplerate=44100):
        """ Function segmentizes a signal into segments of desired length, with a desired shift
        ---
        Arguments:
            segment_length: desired, single segmentlength in seconds -> float
            samplerate:     default=44100, samplerate of signal -> int
        ---
        Returns:
            segments: a list of segment objects
        """
        segment_range = segment_length * samplerate # samplerate shows how many segments is in 1 s
        segments = []
        
        # conversion of shift from [sec] to index range
        if shift == None or shift == 0:
            shift = segment_range
        else:
            shift = int(shift * samplerate)
        
        # initialize the segment limits 
        segment_start = 0
        segment_end = segment_range
        while segment_end <= len(self.signal):
            segment_signal = self.signal[segment_start:segment_end]
            segment = Segment(segment_signal, segment_length, self.id, self.isGood)
            segments.append(segment)
                
            segment_start += shift
            segment_end += shift 
            
        return segments  

class Segment:
    """ an object containing a single sample meant to be prepared as an imput for classifier. 
    ---
    attributes:
        signal:             segment of audio record of the motor during work -> np.array
        lengthSec:          length of the segment in seconds -> float
        motorId:            motor id -> int
        isGood:             motor label -> int
        spec_dB:            spectrogram of the signal, in dB -> 2D np.array
        mel_spec_dB:        melspectrogram of the signal, in dB -> 2D np.array
        mel_spec_dB_flat:   flattened melspectrogram, ready to be saved and stored on hard drive in a proper format
    
    """
    def __init__(self, signal, lengthSec, motorId, isGood):
        self.signal = signal.astype(float)
        self.lengthSec = lengthSec # [s]
        self.motorId = motorId
        self.isGood = isGood
        self.spec_dB = None
        self.mel_spec_dB = None
        self.mel_spec_dB_flat = None
        
    def createSpectrogram(self, n_fft = 2**10, hop_length = 2**10//4):
        """ function creates a spectrogram of a signal 
        ---
        args:
            n_fft:      frame size, needs to be power of 2 eg. 2**12, for more info see help(librosa.sfft) -> int
            hop_length: number of samples between successive frames, for more info see help(librosa.sfft) -> int
        ---
        result:
            self.spec_dB:   a spectrogram of a signal is created and stored in spec_dB attribute -> 2D np.ndarray
        """
        spec = np.abs(librosa.stft(self.signal,
                                   n_fft=n_fft,
                                   hop_length=hop_length,
                                   center=False)) ** 2
        self.spec_dB = librosa.power_to_db(spec)

        
    def createMelSpectrogram(self, n_mels, n_fft=2**10, hop_length=2**10//2, samplerate=44100):
        """ Creates a MelSpectrogram of the given signal
        ---
        args:
            n_mels:     number of Mel bands to generate -> int
            n_fft:      length of the FFT window (should be power of 2 e.g.2**12) -> int
            hop_length: number of samples between successive frames -> int
            samplerate: default=44100, samplerate of the given signal -> int
        ---
        results:
            self.mel_spec_dB: a melspectrogram of a signal is created and stored in mel_spec_dB attribute -> 2D np.ndarray
        """
        mel_spec = librosa.feature.melspectrogram(y=self.signal,
                                                  sr=samplerate,
                                                  n_fft=n_fft,
                                                  hop_length=hop_length,
                                                  n_mels=n_mels)
        self.mel_spec_dB = librosa.power_to_db(mel_spec, ref=np.max)
    
    def displaySpectrogram(self, spec_type="spectrogram", title='Spectrogram', hop_length=2**10//4, samplerate=44100):
        """ displays a spectrogram or melspectrogram
        ---
        args:
            spec_type:  default="spectrogram", other possibility 'mel' -> str
            title:      delfault="Spectrogram", a title of the figure -> str
            hop_length: number of samples between successive frames (should be the same as durig the spectrogram generation) -> int
            samplerate: default=44100, a samplerate of the given signal
        """
        if spec_type == 'mel':
            spec = self.mel_spec_dB
            y_axis = 'mel'
        else:
            spec = self.spec_dB
            y_axis = 'log'
        
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(spec, sr=samplerate,  x_axis='s', y_axis=y_axis,  hop_length=hop_length) #, cmap='Greys');

        plt.colorbar(format='%+2.0f dB');
        plt.title(title);

        plt.show()
        
    def flattenMelSpectrogram(self):
        """ function prepares the melspectrogram to be stored in a dataframe that later will be saved in csv format 
        
        """
        self.mel_spec_dB_flat = self.mel_spec_dB.flatten()


     

def printAllMotorsGraphs(x, motors, motor_feature, domain='time', x_lim=None, y_lim=None, save_pdf=False, pdf_title='pdf_title', y_axis='lin'):
    """
    Prints all the motor signals graphs for desired Motor object attribute (signal, FFT)
    ---
    args:
        - x - x vector shared for all the graphs (samples*step, frequencies)
        - motors - list of Motor objects
        - motor_feature - a Motor object feature that is wanted to be plotted
        - domain - domain of x vector
        - x_lim 
        - y_lim
        - save_pdf
        - pdf_title
        - y_axis - default 'lin', other possibility 'log'
    """

    Y = setY(motors, motor_feature)
    x_label = setDomain(domain) 
    
    
    fig, axs = plt.subplots(len(Y),1, figsize=(15, 75)) 
    fig.subplots_adjust(hspace = 1, wspace=0.1)

    axs = axs.ravel()

    for i, y in enumerate(Y, start=0):

        if y_axis == 'lin': 
            axs[i].plot(x, y)
        elif y_axis == 'log':
            axs[i].semilogy(x, y)
            

        motor_type = "silent motor " if motors[i].isGood else "loud motor "

        axs[i].set_title(f'{motor_type} {i+1}')
        axs[i].grid()
        
        axs[i].set_xlabel(x_label)
        axs[i].set_xlim(x_lim)
        axs[i].set_ylim(y_lim)

    if save_pdf == True:
        fig.savefig(f'./graphs/{pdf_title}.pdf');
        
def setDomain(domain='time'):
    """Sets the proper x_label for plots depending on a domain
        ---
        args:
            domain: default='time', domain for the plot, alternative: 'frequency'
    """
    if domain == 'time':
        label = 'Time t[s]' 
    elif domain == 'frequency':
        label = 'Frequency f[Hz]'
    return label


def setY(motors, motor_feature):
    """ creates a list of desired Motor object attribute for each motor 
    ---
    args:
        motors:         list of Motor objects -> list
        motor_feature:  wanted motor_feature -> str
        
    returns:
        Y:  a list of chosen Motor object attribute
    """
    if motor_feature == 'FFT':
        Y = [motor.FFT for motor in motors]
    else:
        Y = [motor.signal for motor in motors]
    return Y  

def findSegmentByMotorId(segments, motorId):
    """ finds a segment of the motor with chosen Id
    ---
    args:
        segments:   list of all segments -> list
        motorId:    an Id of a chosen motor -> int
    ---
    returns:
        i: an index of a 1st found segment with a chosen motor Id
    """
    for i, segment in enumerate(segments):
        if segment.motorId == motorId:
            return i
    return None    

def compareSpectrograms(segments, compared_motorsIds,
                        n_fft=2**12, n_mels=80, hop_size = 2**12 //4, segments_shift=0, samplerate=44100):
    ''' function displays chosen motors' melspectrograms in 2 columns, in a way that is easy to see the differences between different spectrograms
    ---
    args:
        segments:           list of all segments 
        compared_motorsIds: list of Ids of motors that we want to compare -> list
        n_fft:              length of the FFT window (should be power of 2 e.g.2**12) -> int
        n_mels:             number of Mel bands to generate -> int
        hop_size:           number of samples between successive frames (should be the same as durig the spectrogram generation) -> int
        segments_shift:     default=0, a number we want to add to the index of the 1st segment of the desired motor to show segments from later time of work -> int
        samplerate:         default=44100, a samplerate of the given signal
    
    '''
    n_rows = len(compared_motorsIds) // 2
    
    # Defining the subplots
    fig, ax = plt.subplots(nrows = n_rows, ncols = 2, figsize = (25,10*n_rows))
    ax = ax.ravel() 
    
    for i, motorId in enumerate(compared_motorsIds):
        # finding a proper segment 
        #(segments shift -> for short segments you can choose a segment which is more in the middle of the recorded audio)
        segmentId = findSegmentByMotorId(segments, motorId) + segments_shift
        segment = segments[segmentId] 
        
        # computing a melSpectrogram
        segment.createMelSpectrogram(n_mels=n_mels,
                                     n_fft=n_fft,
                                     hop_length=hop_size)
        
        # display
        librosa.display.specshow(segment.mel_spec_dB,
                                 sr=samplerate,
                                 y_axis = 'mel',
                                 x_axis = 's',
                                 hop_length=hop_size,
                                 ax = ax[i])
        
        # setting title
        motor_label = 'good' if segment.isGood else  'bad'
        title = f'motor {motorId}, {motor_label}'
        ax[i].set_title(title)

def createSegmentsDF(segments, save_csv=False, filename='segments'):
    """ Function creates a pandas DataFrame with samples data needed in further development of a classifier, if needed also saves it in csv format with a given filename
    ---
    args:
        segments:   list of Segment objects -> list
        save_csv:   default=False, indicates the need of saving a DF in a csv format -> bool
        filename:   default='segments', a filename of a saved csv file -> str
    returns:
        df:         pandas dataframe with the first two columns: ["motorId", "isGood"] and flattened melspectrogram in the later columns
    """
    # dataFrame with each melspectrogram metadata
    dfMeta = pd.DataFrame(columns=["motorId", "isGood"])
    # list of list of mel spectrograms
    melSpecs = []
    
    for segment in segments:
        # metadata
        segment_serie = pd.DataFrame({"motorId": [segment.motorId], 
                                   "isGood": [segment.isGood]})
        dfMeta = pd.concat([dfMeta, segment_serie], axis=0, ignore_index=True)
        # melspectrograms
        melSpecs.append(segment.mel_spec_dB_flat)   
        
    
    dfMelSpecs = pd.DataFrame(melSpecs)
    # concatenating DFs horizontally
    df = pd.concat([dfMeta, dfMelSpecs], axis=1)
    
    if save_csv:
        # df.to_csv(f'./preprocessed_data/{filename}.csv')
        df.to_csv(f'/content/drive/My Drive/motor_noise_classifier/preprocessed_data/{filename}.csv')
    
    return df

def saveMelSpecData(shape, n_fft, hop_length, filename="melSpecData"):
    """ saves the shape before flattening, n_fft and hop_length of melspectrogram  to a json file for later reproducton of melspectrograms
    ---
    args:
        shape:      original shape of melspectrogram -> list/tuple
        n_fft:      length of the FFT window -> int
        hop_length: number of samples between successive frames -> int
        filename:   name of the file to be saved -> str
    """
    melSpecShape = {"melSpecShape0": shape[0], 
                    "melSpecShape1": shape[1], 
                    "n_fft": n_fft, 
                    "hop_length": hop_length}
    
    with open(f'/content/drive/My Drive/motor_noise_classifier/preprocessed_data/{filename}.json', 'w') as fp:
        json.dump(melSpecShape, fp,  indent=4)
        
        
    
##################################################################################
# Dimensionality reduction with AutoEncoders and clustering

### Data preparation

def reshapeMelSpecs(df, melSpec_shape):
    """function which reshapes all melspectrograms in the DataFrame back to their actual shape
    Returns a dataframe with 3 columns, first 2 stay the same, the 3rd one contains a reshaped melspectrogram in a 2D np.array form
    ---
    args:
        df:             source dataFrame with first two columns being ['motorId', 'isGood'] the rest is flatenned melspectrogram -> pd.DataFrame
        melSpec_shape:  original shape of the melSpectrogram to be restored -> list/tuple
    ---
    returns:
        df_out:     a DataFrame with first two columns being ['motorId', 'isGood'] and the 3rd one with a melspectrogram with restored original shape
    """        
    # Splitting df into the one with labels 
    # and one with melSpectrograms (and conversion to 2 dimensional np.array with .values)
    df_out = df.iloc[:, :2]
    melSpecs_flat = df.iloc[:, 2:].values
    # reshaping melspecs
    melSpecs = melSpecs_flat.reshape((melSpecs_flat.shape[0], melSpec_shape[0], melSpec_shape[1]))
    print(melSpecs.shape) 


    df_out['melSpec'] = list(melSpecs)
    return df_out

### Visualising MelSpectrograms

def displaySpectrogram(spec, spec_type="spectrogram", title='Spectrogram', hop_length=2**10//4, samplerate=44100):
  """ A function that displays chosen spectrogram/melSpectrogram
  ---
  args:
      spec:         a spectrogram we want to display in a 2D np.array/list form -> np.array/list
      spec_type:    default='spectrogram', can be also 'mel' -> str
      title:        default="Spectrogram", figure title
      hop_length:   default=2**10//4, needs to be a power of 2, see the documentation of librosa.display.specshow -> int
      samplerate:   default=44100, -> int
  
  """
  if spec_type == 'mel':
    y_axis = 'mel'
  else:
    y_axis = 'log'
        
  plt.figure(figsize=(25, 10))
  librosa.display.specshow(spec, sr=samplerate,  x_axis='s', y_axis=y_axis,  hop_length=hop_length) #, cmap='Greys');

  plt.colorbar(format='%+2.0f dB');
  plt.title(title);

  plt.show()


### Data normalisation 

def normalise_dataset(dataset, minimum, maximum):
    """ Function normalises dataset with respect to custom minimum/maximum, which allows to normalise the test set with respect to min/max of the train set
    ---
    args:
        dataset:    dataset in a 2D form, rows are samples (flattened spectrograms) -> 2d np.array
        minimum:    minimum with respect to which the normalisation will be conducted -> float
        maximum:    minimum with respect to which the normalisation will be conducted -> float
    ---
    returns:
        normalised_dataset: dataset after normalisation to (0,1) range -> 2D np.ndarray
    """
    normalised_dataset = (dataset - minimum) / (maximum - minimum)
    return normalised_dataset

def denormalise_dataset(_normalised_dataset, minimum, maximum):
    """ Function denormalise the set back to original values
    ---
    args:
        dataset:    dataset in a 2D form, rows are samples (flattened spectrograms) -> 2d np.array
        minimum:    minimum with respect to which the normalisation will be conducted -> float
        maximum:    minimum with respect to which the normalisation will be conducted -> float
    ---
    Returns:
        denormalised_dataset: dataset in an original range
    """
    datset = normalised_dataset * (maximum - minimum) + minimum
    return dataset

### Building a stacked AutoEncoder

def buildAESubmodel(hidden_layers=1,
                    neurons_in_layers=[1],
                    activations=['sigmoid'],
                    input_shape=(140,44),
                    output_shape=(140,44),
                    submodel_type='encoder'):
  """ function builds an AutoEncoder submodel of users choice (encoder or decoder)
    args:
        hidden_layers:      default=1, number of hidden layers in a submodel -> int
        neurons_in_layers:  default=[1], a list of numbers of neurons in each of the hidden layers -> list(ints)
        activations:        default=['sigmoid], a list of activation functions for each hidden layer -> list(strings)
        input_shape:        default=(140,44), input shape to the 1st layer of the subModel -> tuple
        output_shape:       default=(140,44), input shape to the 1st layer of the subModel -> tuple
        submodel_type:      default='encoder', can be also 'decoder', the type of submodel, defines the choice of input/output layers -> str
  ---
  returns:
    submodel: encoder or decoder of an AE
  """
  assert len(neurons_in_layers) == hidden_layers
  assert len(activations) == hidden_layers
  assert submodel_type in ['encoder', 'decoder']

  # initialising a subModel
  submodel = Sequential()

  if submodel_type == "encoder":
    # flatten input
    submodel.add(Flatten(input_shape=input_shape))
  elif submodel_type == "decoder":
    # add input shape to 1st layer
    submodel.add(Input(shape=input_shape))

  # add hidden layers with activations
  for i in range(hidden_layers):
    submodel.add(Dense(neurons_in_layers[i], activation=activations[i]))

  if submodel_type == 'decoder':
    # add output layer
    submodel.add(Dense(output_shape[0] * output_shape[1], activation="sigmoid"))
    # reshape the output back to original shape
    submodel.add(Reshape(output_shape))
    

  return submodel


def buildModel(submodels, 
                loss='mse', 
                optimizer='adam', 
                metrics=['mae']):
  """Function joins submodels in one bigger model and compiles it
  ---
    args:
        submodels:  a list of submodels -> list(keras.model objects)
        loss:       default='mse', a loss function for training -> str
        optimizer:  default='adam' -> str
        metrics:    default=['mae'] -> list(strs)
  ---  
    Returns:
        model: completely built and compiled model
  """
  model = Sequential(submodels)
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)
  return model

# reseting a model
def reinitialiseModel(model):
    """ Function reinitialises model to reset all the weights
    Function is a modified function written by billbradley here: https://github.com/tensorflow/tensorflow/issues/48230
    ---
    args:
        model:  model to be reinitialised -> keras.model
    
    """
    weights = []
    initializers = []
    for submodel in model.layers:
        # print(submodel)
        if isinstance(submodel, keras.Sequential):
            reinitialiseModel(submodel)
        else:
            layer = submodel
            if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
                weights += [layer.kernel, layer.bias]
                initializers += [layer.kernel_initializer, layer.bias_initializer]
            elif isinstance(layer, keras.layers.BatchNormalization):
                weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
                initializers += [layer.gamma_initializer,
                                layer.beta_initializer,
                                layer.moving_mean_initializer,
                                layer.moving_variance_initializer]
            elif isinstance(layer, keras.layers.Embedding):
                weights += [layer.embeddings]
                initializers += [layer.embeddings_initializer]
            elif isinstance(layer, (keras.layers.Reshape, 
                                    keras.layers.MaxPooling2D, 
                                    keras.layers.Flatten, 
                                    keras.layers.Dropout)):
                # These layers don't need initialization
                continue
            else:
                raise ValueError('Unhandled layer type: %s' % (type(layer)))
    for w, init in zip(weights, initializers):
        w.assign(init(w.shape, dtype=w.dtype))

def KfoldValidationAETraining(model,
                              X_train,
                              k_folds = 1,
                              epochs=100,
                              batch_size=16,
                              callbacks=None):
  """Function performs a KFold validation on a chosen model, calculates the mean MAE error and rounded mean number of epochs before starting overfitting
  ---
  args:
      model:        keras.model object  to get trained in a KFold validation method -> keras.model object (autoencoder)
      X_train:      training set -> 2d np.array
      k_folds:      default=1, number of k folds for validation process -> int
      epochs:       default=100, number of training epochs -> int
      batch_size:   default=16, should be power of 2 -> int
      callbacks:    default=None, a list of callbacks -> list
  ---
  returns: 
    mean_MAE_error: mean mean absolute error of all the trainings
    epochs_number:  rounded mean number of epochs at whicj trainings were finished
  """

  num_val_samples = len(X_train) // k_folds
  all_scores = []
  all_epochs = []

  for i in range(k_folds):
    # print(f"fold: #{i+1}")
    X_val = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    partial_X_train = np.concatenate(
        [X_train[:i * num_val_samples], 
         X_train[(i + 1) * num_val_samples:]], 
        axis=0)
    # reset model
    reinitialiseModel(model)

    history = model.fit(partial_X_train, 
                        partial_X_train,
                        epochs=epochs,
                        validation_data=[X_val, X_val],
                        batch_size=batch_size,
                        callbacks=callbacks, 
                        verbose=0)
    # getting number of epochs
    # epochs_at_finish = len(history.history['loss'])
    epochs_at_min_loss = np.argmin(history.history['val_mse'])
    
    # val_mse, val_mae = model.evaluate(X_val, X_val, verbose=0)
    val_mae = np.min(history.history['val_mae'])
    
    # MAE being the most intuitive loss 
    all_scores.append(val_mae)
    all_epochs.append(epochs_at_min_loss)

  return np.mean(all_scores), np.ceil(np.mean(all_epochs))


### Hyperparameter optimization
def GridSearchAE(params, X_train, k_folds=2, epochs=100, callbacks=None, melSpec_shape=(140,44)):
    """ Function performs a grid search for autoencoders with chosen parameters, checking their mae error results
    ---
    args: 
        params: a dataFrame containing all possible combinations of parameters to get checked,
                should have columns=['AE_architecture', 'activation', 'loss', 'batch_size'] -> DataFrame
        X_train: training dataset -> 2D np.array
        k_folds: default=2, number of k folds -> int
        epochs: default=100 -> int
        callbacks: default=None, a list of callbacks -> list
        melSpec_shape: default=(140,44), needed for input/output layers definition -> tuple
    ---
    Returns: 
    results: a dataframe with a structure: 
             columns = ['encoder_layers', 'decoder_layers', 'loss','activation', 'batch_size', 'epochs_nb', 'MAE']
    """ 
    total_nb_of_models = len(params.activation)

    results = pd.DataFrame(columns=['encoder_layers', 'decoder_layers', 'loss', 'activation', 'MAE', 'epochs_nb'])
    for i, model_params in params.iterrows():
        # iterations counter
        print(f'#{i+1}/{total_nb_of_models}')
        AE_arch = model_params.AE_architecture
        # Encoder
        hidden_layers_enc = len(AE_arch)//2 + 1
        neurons_enc = AE_arch[:hidden_layers_enc]
        encoder = buildAESubmodel(hidden_layers=hidden_layers_enc,
                                neurons_in_layers=neurons_enc,
                                activations=[model_params.activation for i in range(hidden_layers_enc)],
                                submodel_type='encoder',
                                input_shape=melSpec_shape)
        # Decoder
        hidden_layers_dec = len(AE_arch)//2
        neurons_dec = AE_arch[hidden_layers_enc:]
        decoder = buildAESubmodel(hidden_layers=hidden_layers_dec,
                                neurons_in_layers=neurons_dec,
                                activations=[model_params.activation for i in range(hidden_layers_dec)],
                                submodel_type='decoder',
                                input_shape=neurons_enc[-1],
                                output_shape=melSpec_shape)
        # building AutoEncoder
        autoencoder = buildModel([encoder, decoder],
                              loss=model_params.loss)
    
        # training a model
        MAE, best_epochs_nb = KfoldValidationAETraining(autoencoder, 
                                                        X_train, 
                                                        k_folds = k_folds, 
                                                        epochs=epochs, 
                                                        batch_size=model_params.batch_size, 
                                                        callbacks=callbacks)
        # saving results to pandas DataFrame
        result = {'encoder_layers': neurons_enc,
                  'decoder_layers': neurons_dec, 
                  'loss': model_params.loss, 
                  'activation': model_params.activation, 
                  'batch_size': model_params.batch_size,
                  'epochs_nb': best_epochs_nb, 
                  'MAE': MAE}
        results = results.append(result, ignore_index=True)
            
    return results


#### Feature Extraction
def ReducedFeaturesHistograms(features_reduced, mins, maxes, motor_label='Good', y_lim=[0,1]):
  """Function displays histograms of all the reduced features which enables their distribution observation
  ---
  args:
      features_reduced:     result of reducing a samples by a encoder -> 2d np.array
      mins:                 list of minimal values of all the samples for each feature -> float
      max:                  list of maximum values of all the samples for each feature -> float
      motor_label:          motor label for a main figure title -> str
      y_lim:                default=[0,1], limits for y axis -> list
  """

  nb_of_subplots = features_reduced.shape[1]
  fig, axs = plt.subplots(ncols=3, nrows=int(np.ceil(nb_of_subplots/3)), figsize=(15, nb_of_subplots))
  fig.suptitle(f"{motor_label} motors", fontsize=18, y=0.95)
  plt.subplots_adjust(hspace=0.5)

  feature_ids = np.arange(nb_of_subplots)

  for feature_id, ax, minimum, maximum in zip(feature_ids, axs.ravel(), mins, maxes):
    # if range == 0, extend the boundries
    if minimum ==  maximum:
      minimum -= 0.5
      maximum += 0.5
      
    data = features_reduced[:, feature_id]  
    ax.hist(data, bins=50, weights=np.ones(len(data)) / len(data), range=(minimum, maximum))
    # chart formatting
    ax.set_ylim(y_lim)
    ax.set_title(f'feature: {feature_id}')

  plt.show()


def ExtractNotableFeatures(maxes_X_reduced, mins_X_reduced, threshold=0):
    """Function extracts the notable features, important for recognising if the motor is 
    good or not, by ommiting the features that are always the same, 
    both for good and bad motors. It does it by checking the range of the distribution of the feature
    ---
    args:
        maxes_X_reduced:    list of maximums for each of the features 
                            resulting from passing a sample through encoder -> list(floats)
        mins_X_reduced:     list of minimums for each of the features 
                            resulting from passing a sample through encoder -> list(floats)
        threshold:          default=0, a threshold of the range below which the feature is 
                            treated as NOT important/notable -> float
    ---
    Returns:
        notable_features:   list of notable features, imporant in motor noise classification
    """
    notable_features = []
    ranges_X_reduced = maxes_X_reduced - mins_X_reduced
    for i, feature_range in enumerate(ranges_X_reduced):
        if abs(feature_range) > threshold:
            notable_features.append(i)

    return notable_features

### TSNE visualisation 
def printMotorIdAnnotations(nb_of_motors, data_reduced_2D):
    """ Function prints motors id on a TSNE visualisatin scatterplot
    ---
    args:
        nb_of_motors:       total number of motors to be shown -> int
        data_reduced_2D:    dataframe with features reduced in TSNE -> npd.DataFrame
    """
    for motorId in range(1, nb_of_motors):
        X_red = data_reduced_2D[data_reduced_2D.motorId == motorId]
        # ax.scatter(X_red.X_red0, X_red.X_red1, s=2)
  
        # taking one point for annotation
        sample_point = X_red.sample(1)
        plt.text(sample_point.X_red0+1.5, sample_point.X_red1+1.5, motorId, fontsize=16)

### Clustering
def SearchOptimalK(X, i_max=12):
    """Function performs a KMean clustering for number of  cluster numbers and then plots the k_number - inertia_ plot which  enables an elbow observation
    ---
    args:
        X:      dataset -> 2d np.arrays
        i_max:  default=12 a maximum number of clusters to get checked
    """
    
    kMean_search_inertias = []
    for i in range(2, i_max):
      kmeans = KMeans(n_clusters=i)
      clusters = kmeans.fit(X)
      kMean_search_inertias.append(clusters.inertia_)
    
    plt.figure()
    plt.plot(np.arange(2, i_max), kMean_search_inertias, 'x-')
    plt.ylabel('inertia')
    plt.xlabel('nb of clusters')
    plt.grid(which='minor')
    plt.show()

### Heatmap
def PrepareClusterHeatmapDF(motorIds_total, clusters):
  """Function calculates how many samples for each motor are in which cluster and builds a dataframe with a structure ready to b shown on a heatmap
  ---
  args:
      motorsIds_total:  list of motors Ids for each sample (after shuffling) -> list(ints)
      clusters:         result of kMean clustering -> list(ints)
  ---
  returns:
    count_table_df: Dataframe with count of how many samples of each motor were put in each cluster, columns=[f'cluster{i}' for i in range(k)], rows = motor ids
  """
  nb_of_motors = np.max(motorIds_total)
  # nb of clusters
  k = np.max(clusters) + 1
  
  count_table = np.zeros([nb_of_motors, k])
  for motorId, cluster in zip(motorIds_total, clusters):
    count_table[motorId-1, cluster] += 1
  
  count_table_df = pd.DataFrame(count_table, columns=[f'cluster{i}' for i in range(k)], index=np.arange(1, nb_of_motors+1))
  return count_table_df


### Changing labels
def ChangeLabels(data_src, motors_to_change_labels):
    """Function changes the labels of chosen motors,
    ---
    args:
        data_src:                   dataframe with flattened spectrograms, ready to be saved in csv format
        motors_to_change_labels:    list of motorIds of motors which labels were wrong 
                                    (decision made after observing a clustering) -> list(ints)
    ---
    Returns: 
        data_changed: dataFrame with changed labels in a ready-to-save form (melspectrogram flattened)
    """
    data_changed = data_src.copy()

    for motorId in motors_to_change_labels:
      old_label = data_changed.isGood[data_changed.motorId == motorId].sample().item()
      new_label = True if old_label == False else False
      data_changed.loc[data_changed.motorId == motorId, 'isGood'] = new_label

    return data_changed  
    
###################################################################
# CNN model

def splitTrainTestByMotorId(df, nb_of_test_motors=5, custom_shuffled_list=[], threshold=0.65, print_results=False):
    """ Function that splits a dataset into 2 subsets with respect to motor ids, if the user doesnt use a custom list of motors, the function takes care of creating one by shuffling the list of motors from the given dataset. It also takes care about the balance of the smaller of the sets, the labels True/False ratio will not be greater than the given threshold
    ---
    args:
        df:                     dataframe with a dataset to be splitted 
                                (columns=['motorId', 'isGood', 'melSpec']) -> pd.DataFrame
        nb_of_test_motors:      number of motors to be left in the smaller of the sets after split -> int
        custom_shuffle_list:    custom shuffled list of motors given by the user 
                                (can be used for tests or to reproduce the results from previous experiments)
                                If left empty, then the list of shuffled motors is generated by the function -> list
        threshold:              default=0.65, Maximum ratio of labels True/False imbalance 
                                in the smaller set -> float (range 0 to 1)
        print_results:          default=False, change to True if you want to see the results of the split printed -> bool
    ---
    returns:
        df_train:                   bigger of the datasets after split 
        df_test:                    smaller of the datasets after split
        shuffled_list_of_motors:    list of motor Ids used to split the dataset
    
    """
    nb_of_motors = df.motorId.max()

    # shuffle the list of motors, the 1st 'nb_of_test_motors' will
    # go to the test set, repeat the shuffling if the good/bad motors ratio
    # is too high (>=65%)

    testset_isBalanced = False

    while not testset_isBalanced:
        # check if user wants to use his/her own shuffled list
        if len(custom_shuffled_list) == 0:
            shuffled_list_of_motors = shuffle(df.motorId.unique())
        else:
            shuffled_list_of_motors = custom_shuffled_list
            testset_isBalanced = True
            break
        
        # check if the dataset is balanced
        testset_isBalanced = checkTestsetBalance(df, shuffled_list_of_motors, nb_of_test_motors, threshold)

    if print_results == True:
        print(f'shuffled_list_of_motors {shuffled_list_of_motors}')
        print(f'trainSet {shuffled_list_of_motors[nb_of_test_motors:]}')
        print(f'testSet {shuffled_list_of_motors[:nb_of_test_motors]}')

    df_test = df.loc[df.motorId.isin(shuffled_list_of_motors[:nb_of_test_motors])].copy()
    df_train = df.loc[df.motorId.isin(shuffled_list_of_motors[nb_of_test_motors:])].copy()

    return df_train, df_test, shuffled_list_of_motors

def checkTestsetBalance(df, shuffled_list_of_motors, nb_of_test_motors, threshold):
    """ function checks if the split for shuffled list is balanced (the labels True/False ratio is smaller then given threshold)
    ---
    args:
        df:                         dataframe with a dataset to be splitted 
                                    (columns=['motorId', 'isGood', 'melSpec']) -> pd.DataFrame
        shuffled_list_of_motors:    list of motors used for split -> list
        nb_of_test_motors:          number of motors that will be left in the smaller dataset after split
        threshold:                  maximum ratio of labels True/False imbalance in the smaller set -> float (range 0 to 1)
    ---
    returns:
        isBalanced:     an indicator showing if the split is balanced or not
    """
    good_motors_count = 0
    bad_motors_count = 0
    for i in range(nb_of_test_motors):
        motorId = shuffled_list_of_motors[i]
        motor_label = df.isGood.loc[df.motorId == motorId].sample().bool()
        if motor_label == True:
            good_motors_count +=1
        else:
            bad_motors_count +=1 

    ratio = np.max([good_motors_count, bad_motors_count]) /  nb_of_test_motors
    if ratio >= threshold:
        isBalanced = False
    else:
        isBalanced = True
    return isBalanced
    
### Building a model

    
def buildCNNInputSubmodel(input_shape):
    """ Function builds a CNN input submodel including a Reshape layer with a proper input_shape, it adds a channel dimension to the input data
    ---
    args:
        input_shape: shape of input data -> list/tuple
    ---
    returns:
        input: input submodel of a CNN
    """
    input = Sequential(name='input_submodel')
    input.add(Reshape([input_shape[0], input_shape[1], 1], input_shape=input_shape))

    return input

def buildCNNMiddleSubmodel(nb_of_conv_layers=1,
                           filter_numbers=[32],
                           kernel_sizes=[3],
                           activations=['selu'],
                           dropouts=None,
                           ):
    """ Function builds a middle part of CNN including all the Convolutional, maxPooling and Dropout layers
    ---
    args:
        nb_of_conv_layers:  number of convolutional layers of the submodel -> int
        filter_numbers:     list of filters numbers for each convolutional layer -> list
        kernel_sizes:       list of kernel sizes for each convolutional layer -> list
        activations:        list of activation functions for each convolutional layer -> list
        dropouts:           list of droputs ratios for each convolutional layer 
                            (if None then there is no dropout layer for the convolution) -> list
    ---
    returns:
        submodel:           a middle submodel of the CNN
    """
    assert nb_of_conv_layers == len(kernel_sizes)
    assert nb_of_conv_layers == len(filter_numbers)
    assert nb_of_conv_layers == len(activations)

    if dropouts == None:
        dropouts = [None for i in range(nb_of_conv_layers)]

    submodel = Sequential(name='middle_submodel')

    for i in range(nb_of_conv_layers):
        submodel.add(Conv2D(filter_numbers[i], 
                                   kernel_size=kernel_sizes[i], 
                                   padding="same", 
                                   activation=activations[i]))
        submodel.add(MaxPool2D(pool_size=2))
        if dropouts[i] != None:
            submodel.add(Dropout(dropouts[i]))

    # flatten the output of submodel
    submodel.add(layers.Flatten())

    return submodel
    
###

def convertDatasetToArrays(df, return_motorIds_array=False):
    """ Function shuffles and splits the features from the labels and motor ids and return them in seperate np.ndarrays
    ---
    args:
        df:                     dataset -> pd.DataFrame
        return_motorIds_array:  default=False, set to True if you wanto motorIds returned 
                                as a np.ndarray returned on the 3rd position -> bool
    ---    
    returns:
        X:          features matrix
        y:          label array
        motorIds:   motorIds array (returned only if return_motorIds_array=True)
    """
    X = np.array(list(df.melSpec))
    y = np.array(df.isGood).astype('int')
    motorIds = np.array(df.motorId)

    X, y, motorIds = shuffle(X, y, motorIds)

    if return_motorIds_array:
        return X, y, motorIds
    else:
        return X, y
        

def KfoldValidationCNNTraining(model,
                               df_train,
                               k_folds = 1,
                               epochs=100,
                               batch_size=16,
                               callbacks=None,
                               shuffled_motor_list=[],
                               return_histories=False):
    """ Function performs a Kfold validation on the given model and dataset
    ---
    args:
        model:      model to be tested in Kfold validation -> keras.model
        df_train:   dataset used in Kfold validation -> d.DataFrame
        k_folds:    default=1, number of k folds to be performed -> int
        epochs:     default=100, maximum number of training epochs -> int
        batch_size: defaulr=16, size of the training batch, should be power of 2 -> int
        callbacks:  default=None, list of callbacks used in training the CNN -> list
        shuffled_motor_list:    default=[], custom shuffled list of motors given by the user 
                                (can be used for tests or to reproduce the results from previous experiments)
                                If left empty, then the list of shuffled motors is generated by the function -> list
        return_accuracy_list:   default=False, set to True if you want all the scores returned in the list -> bool
    returns:
        mean_accuracy:  mean mean absolute error of all the trainings
        epochs_number:  rounded median number of epochs at which trainings were finished
        accuracies:     all accuracies, returned only if return_accuracy_list==True
    """

    if len(shuffled_motor_list) == 0:
        all_train_motors = shuffle(df_train.motorId.unique())
    else:
        all_train_motors = shuffled_motor_list.copy() 

    nb_val_motors = len(all_train_motors) // k_folds

    all_scores = []
    all_epochs = []
    all_histories = []
    for i in range(k_folds):
        # animate the fold counter
        sys.stdout.write('\r'+ f"fold: #{i+1}")
        # print(f"fold: #{i+1}")
        # print(f"TestSet: {all_train_motors[:nb_val_motors]}")
        # split train and validation sets
        partial_df_train, df_val, _ = splitTrainTestByMotorId(df_train,
                                                           nb_of_test_motors=nb_val_motors,
                                                           custom_shuffled_list=all_train_motors,
                                                           threshold=1.0)
        partial_X_train, partial_y_train = convertDatasetToArrays(partial_df_train)
        X_val, y_val = convertDatasetToArrays(df_val)

        # move the current validation motors to the end of the list
        # it's necessary becaue of splitTrainTestByMotorId function construction
        # it takes first n motors from the list to test/validation set
        temp = np.copy(all_train_motors[:nb_val_motors])
        all_train_motors[:-nb_val_motors] = np.copy(all_train_motors[nb_val_motors:])
        all_train_motors[-nb_val_motors:] = temp
        
        # reset model
        reinitialiseModel(model)

        history = model.fit(partial_X_train, 
                            partial_y_train,
                            epochs=epochs,
                            validation_data=[X_val, y_val],
                            batch_size=batch_size,
                            callbacks=callbacks, 
                            verbose=0)
        # getting number of epochs
        # epochs_at_finish = len(history.history['loss'])
        epochs_best_accuracy = np.argmax(history.history['val_accuracy']) + 1
        all_epochs.append(epochs_best_accuracy) 

        # getting model Accuracy
        # val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        val_accuracy = np.max(history.history['val_accuracy'])
        all_scores.append(val_accuracy)
        
        # adding history to histories list
        all_histories.append(history)

    print(f'\naccuracies: {all_scores}, mean: {np.mean(all_scores)}\nepochs: {all_epochs}, median: {np.median(all_epochs)}')
    if return_histories:
        return np.mean(all_scores), int(np.ceil(np.median(all_epochs))), all_histories
    else:    
        return np.mean(all_scores), int(np.ceil(np.median(all_epochs)))


### Grid Search
def GridSearchCNN(params, 
                  df_train, 
                  k_folds=4, 
                  epochs=20, 
                  callbacks=None, 
                  input_shape=(144,44)):
    """ function performs a grid search for given model parameters (by doing Kfold validation for each parameters set) 
        returning a table with results
    ---
    args:
        params:     dataframe with parameters to get checked, 
                    columns=['nb_of_layers', 'filter_numbers', 'activation', 'batch_size', 'dropouts']
                    -> pd.DataFrame
        df_train:   dataset used to train the model -> pd.DataFrame
        k_folds:    default=4, number of k_folds to be done durng cross validation
        epochs:     default=100, maximum number of training epochs -> int
        callbacks:  default=None, list of callbacks used in training the CNN -> list
        input_shape: input data shape -> tuple/list
    ---
    returns:
        results:    table containing all the checked parameters together with the obtained results 
                    (accuracies and mean epochs numbers)
    
    """

    # shuffling a list for train/val set split
    all_train_motors = shuffle(df_train.motorId.unique())

    total_nb_of_models = len(params.activation)

    results = pd.DataFrame(columns=['nb_of_layers', 'filter_numbers', 'activation', 'batch_size', 'dropouts','epochs_nb', 'accuracy'])
    for i, model_params in params.iterrows():
        # iterations counter
        print(f'#{i+1}/{total_nb_of_models}')
        # building submodels
        input_submodel = buildCNNInputSubmodel(input_shape)

        nb_of_conv_layers = len(params.filter_numbers[i])
        middle_submodel = buildCNNMiddleSubmodel(nb_of_conv_layers=len(model_params.filter_numbers),
                                                 filter_numbers=model_params.filter_numbers,
                                                 kernel_sizes=[3 for i in range(nb_of_conv_layers)],
                                                 activations=[model_params.activation for i in range(nb_of_conv_layers)],
                                                 dropouts=model_params.dropouts
                                                 )
        output_submodel = Dense(1, activation='sigmoid', name='output')

        model = buildModel([input_submodel, middle_submodel, output_submodel], 
                            loss='binary_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy'])
        # training a model
        accuracy, best_epochs_nb = KfoldValidationCNNTraining(model,
                                                              df_train, 
                                                              k_folds=k_folds, 
                                                              epochs=epochs, 
                                                              batch_size=model_params.batch_size, 
                                                              callbacks=callbacks,
                                                              shuffled_motor_list=all_train_motors)
        # saving results to pandas DataFrame
        result = {'nb_of_layers': nb_of_conv_layers,
                  'filter_numbers': model_params.filter_numbers,
                  'activation': model_params.activation, 
                  'batch_size': model_params.batch_size,
                  'dropouts': model_params.dropouts,
                  'epochs_nb': best_epochs_nb, 
                  'accuracy': accuracy}
        results = results.append(result, ignore_index=True)
            
    return results

def formGridSearchCatplot(grid_search_results, parameter_to_visualise='nb_of_layers', ylim=[0.7, 1.05]):
    """ Function visualises GridSearch results by plotting a categorical plot for chosen parameter
    ---
    args:
        grid_search_results:    results of grid search -> pd.DataFrame
        parameter_to_visualise: default='nb_of_layers', one of the parameters checked in grid 
                                search that the user wants to visualise on the categorical plot -> str
        ylim:                   default=[0.7, 1.05] -> list/tuple
    """
    x=parameter_to_visualise
    y='accuracy'

    plt.figure();
    sns.catplot(x=x, y=y, data=grid_search_results, s=8, height=8, aspect=1.25)
    plt.grid()
    plt.ylim(ylim)
    plt.show()
    
    
### Making predictions and processing results
def countPredictionsByMotorId(y, preds, motorIds):
    """ Function creates a dataframe containing counts of predictions for each motor, 
        the sumed up prediction and its confidence in percents
    ---
    args:
        y:          ground truth labels array -> np.ndarray
        preds:      model predictions   -> np.ndarray
        motorIds:   array of model Ids for each label/prediction -> np.ndarray
    ---
    returns:
        df_motors:  a dataframe with counts of predictions for each motor, 
                    the sumed up prediction and its confidence in percents
    """
    df_samples = pd.DataFrame(data={'motorId': motorIds, 'label': y, 'prediction': preds})

    # counting predictions by motorId groups
    df_motors_good = df_samples.loc[df_samples.prediction==1].groupby('motorId').prediction.count()
    df_motors_bad = df_samples.loc[df_samples.prediction==0].groupby('motorId').prediction.count()

    # extracting a ground truth label for each motor
    motorIds = np.unique(motorIds)
    labels = []
    for motorId in motorIds:
        label = df_samples.label.loc[df_samples.motorId==motorId].sample()
        labels.append(bool(int(label)))

    df_motors = pd.DataFrame(data={'good': df_motors_good, 
                                   'bad': df_motors_bad})
    df_motors = df_motors.replace(np.nan, 0)
    df_motors['ground_truth'] = labels

    # make summed up predictions and calculate their confidence
    df_motors["summed_prediction"] = df_motors.good >= df_motors.bad 
    df_motors["prediction_confidence"] = df_motors[['good', 'bad']].max(axis=1) / (df_motors.good+df_motors.bad)
    
    return df_motors

#
def finalCrossValidationCNN(model,
                            dataset,
                            nb_test_motors=10,
                            iter_nb=1,
                            epochs=5,
                            batch_size=16):
    """ Function performs a final cross validation on the given model and full dataset, it does not include any validation during training 
        and calculates the accuracies with respect to whole data for the motor
    ---
    args:
        model:      model to be tested  -> keras.model
        df_train:   full dataset  -> d.DataFrame
        folds:      default=1, number of folds to be performed -> int
        epochs:     default=5,  number of training epochs -> int
        batch_size: defaulr=16, size of the training batch, should be power of 2 -> int
        shuffled_motor_list:    default=[], custom shuffled list of motors given by the user 
                                (can be used for tests or to reproduce the results from previous experiments)
                                If left empty, then the list of shuffled motors is generated by the function -> list
    
    returns:
        mean_accuracy:  mean mean absolute error of all the trainings
        accuracies:     all accuracies
    """ 

    all_scores = []
    all_tables = []
    for i in range(iter_nb):
        print(f"fold: #{i+1}")
        # print(f"TestSet: {all_train_motors[:nb_test_motors]}")
        # split train and test sets
        partial_df_train, df_test, _ = splitTrainTestByMotorId(dataset,
                                                                nb_of_test_motors=nb_test_motors,
                                                                threshold=0.65)
        partial_X_train, partial_y_train = convertDatasetToArrays(partial_df_train)
        X_test, y_test, motorIds_test = convertDatasetToArrays(df_test, return_motorIds_array=True)
        
        # reset model
        reinitialiseModel(model)

        history = model.fit(partial_X_train, 
                            partial_y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=1
                            )
        # getting model Accuracy
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test accuracy (general, NOT w.r.t. motors!): {test_accuracy}')

        # making predictions for current test subset
        preds = model.predict(X_test)
        preds_test_round = []
        for pred in preds:
            pred_round = 1 if pred >= 0.5 else 0
            preds_test_round.append(pred_round)
        # forming a count table for the results w.r.t. the motors
        results_table = countPredictionsByMotorId(y_test, preds_test_round, motorIds_test)
        all_tables.append(results_table)
        # calculating accuarcies for final full predictions w.r.t motor
        final_accuracy = (results_table.ground_truth == results_table.summed_prediction).astype(int).sum() / len(results_table.ground_truth)
        print(f"Test accuracy (W.R.T motors): {final_accuracy}")
        all_scores.append(final_accuracy)
    
    mean_accuracy = np.mean(all_scores)

    return mean_accuracy, all_scores, all_tables