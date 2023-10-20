import logging
import os
from c_config import B_API_KEY, B_API_SECRETKEY, T_API_KEY, chat_id  # Fill-up the c_config file with your keys

#--- Configuration parameters ---

loc_folder = ''  # I/O folder can be set here (Ex:loc_folder = 'C:/Users/USER/PycharmProjects/ProjectA'//). An empty string will look for resources in the root folder.
crypto = ['BTC/TUSD']  # Market pair chosen for trading
target_name = "Return"  # name of the target column
interval = '1s'  # Sample or Candle size of the imported dataset
future_points = 8  # Once the dataset is resampled this value defines how far in the future are we trying to predict (Ex: If resample size = 10 seconds and future_points = 18, the program is predicting at 180 seconds into the future)
resample_co = 15  # Resample size expressed as a number of intervals
time_steps = 75 * future_points  # "future_points" will be used as a base to calculate the sequence length of the LSTM RNN
sequences = 30  # This value represents the number of sequences that are being used in order to train the LSTM.
dense_units = 2  # Dense layer units, this should stay at 1.
dataset_name = 'candles.csv'  # Name of the data set that will be shared and updated constantly by the modeler and trader tools
hyper_loops = 100  # Number of random hyperparameter combinations to try for each combination of indicators that the tuner uses.
comb_loops = 50  # Number of random indicator combinations to try for each combination of hyperparameters that the tuner uses.
desc = 'A'  # Add a description to the output files of the modeler in case we need to save different models
validations = 5  # When modeler_mode is set to 'parallel' the model will calculate the accuracy N times with the same hyperparameters and set the score based on this average in order to get a more consistent value by train/test the model at different times.
#arr_list = ['so_data', 'ema_data', 'rsi_data']  # Once the best indicators have been found by the tuner and in order to use that combination in the modeler is necessary to update this list accordingly with that combination first.
model_reset = 14400  # Time expressed in seconds after which the modeler is reset to find a new model in order to keep the model updated.
modeler_mode = 'solo'  # Set if the modeler will work by itself in "solo" mode or in "parallel" mode with the trader to keep the model updated.
test_size = 0.3  # Fraction of the dataset that will be reserved for testing during modeling
asset_precision = 5
threshold_time = 21600  # A dataset will be built and constantly updated from the websocket seconds-candle stream of prices. This value sets a limit expressed in seconds on the length of this dataset
best_thres = 0  # The trader will use this value to filter weak predictions below a certain absolute value defined by this variable so that orders are only executed when signals are strong enough.
verbose = 0  # Level of description while executing model methods.

# Hyperparameter Tuner Configuration
'''This tool can be set in three different ways depending on the "hyper_mode" value: 
hyper_mode = 0 --> Search for the best hyperparameters randomly in a number of combinations of indicators defined by comb_loops.
hyper_mode = 1 --> Search for the best hyperparameters randomly only in the combination of indicators defined in arr_list.
hyper_mode = 2 --> Search for the best hyperparameters randomly on all combinations of indicators.
hyper_mode = 3 --> Uses the default hyperparameter values on all random combinations of indicators.
'''
hyper_mode = 1  # The Hyperparameter finder can try different hyperparameters on all the indicators at the same time or try different Hyperparamenter on all combinations of

# Indicator list for the tuner, modeler, and trader:
arr_list = ['so', 'pvo', 'ema', 'rsi', 'srsi', 'cci', 'psar', 'vwap']  # Once the best indicators have been found by the tuner and in order to use that combination in the modeler is necessary to update this list accordingly with that combination first.
#arr_list = []
# No indicatorsLayers: 10 Epochs: 90 Batch Size: 15 Dropout Rate: 0.2 Learning Rate: 1e-06 Input Act. Func: tanh Output Act. Func: tanh

# Default hyperparameters for the Tuner and Modeler (If hyper_mode = 2 the tuner will use these):
layers = 128
dropout = 0.2
learning_rate = 0.000001
act_input = 'sigmoid'
act_output = 'tanh'
epochs = 50
batch_size = 25

# Logger Configuration

file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logger.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M', filename=file, filemode='a')
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)
logger = logging.getLogger('Log')

# Files config
model_name = f'model-{desc}.h5'
scalers_name = f'scalers-{desc}.pkl'
