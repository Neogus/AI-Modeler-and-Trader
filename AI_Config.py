import logging
import os
from c_config import B_API_KEY, B_API_SECRETKEY, T_API_KEY, chat_id  # Fill-up the c_config file with your keys

loc_folder = ''  # I/O folder can be set here (Ex:loc_folder = 'C:/Users/USER/PycharmProjects/ProjectA'//). An empty string will look for resources in the root folder.
crypto = ['BTC/TUSD']  # Market pair chosen for trading
target_name = "Return"  # name of the target column
interval = '1s'  # Sample or Candle size of the imported dataset
conv_to_interval = 10  # Resample size expressed as a number of intervals
resample_size = f'{str(conv_to_interval)}s'
future_points = 18  # Once the dataset is resampled this value defines how far in the future are we trying to predict (Ex: If resample size = 10 seconds and future_points = 18, the program is predicting at 180 seconds into the future)
time_steps = 10 * future_points  # future points will be used as base to calculate the time length or sequence length of each sequence in the LSTM RNN
sequences = 50  # This value represents the number of sequences that are being used in order to train the LSTM.
dense_units = 1  # Dense layer units, this should stay at 1.
dataset_name = 'candles.csv'  # Name of the data set that will be shared and updated constantly by the modeler and trader tools
desc = 'A'  # Add a description to the output files of the modeler in case we need to save different models
validations = 3  #  When modeling this value represents the number of loops in order to get a mean and a deviation.
arr_list = ['so_data', 'pvo_data', 'ema_data', 'rsi_data', 'srsi_data', 'cci_data', 'psar_data', 'vwap_data']  # Once the best indicators have been found by the tuner and in order to use that combination in the modeler is necessary to update this list accordingly with that combination first.
model_reset = 14400  # Time expressed in seconds after which the modeler is reset to find a new model in order to keep the model updated.
modeler_mode = 'solo'  # Set if the modeler will work by itself on "solo" mode or in "parallel" mode with the trader to keep the model udpated.
test_size = 0.25  # Fraction of the dataset that will be reserve for testing during modeling
asset_precision = 5
threshold_time = 21600  # A dataset will be built and constantly updated from the websocket seconds-candle stream of prices. This values sets a limit expreseed in seconds on the length of this dataset
best_thres = 0  # The trader will use this value to filter weak predictions below a certain absolute value defined by this variable so that orders are only exectued when signals are strong enough.

# Hyperparameter Tuner Configuration
'''This tool can be set in three different ways depending on the "hyper_mode" value: 
hyper_mode = 0 --> Search for the best hyperparameters in each combinations of indicators
hyper_mode = 1 --> Search for the best hyperparameters in the last combination of indicators only which contains all indicators in the arr_list
hyper_mode = 2 --> Search for the best combination of indicators using pre-defined hyperparameter values. When finding the best combination in order to use it the arr_list should be modified to include only these indicators.
'''
hyper_mode = 1  # The Hyperparameter finder can try different hyperparameters on all the indicators at the same time or try different Hyperparamenter on all combinations of

# Default hyperparameters (If hyper_mode = 2 the tuner will use these):
layers = 8
dropout = 0.3
learning_rate = 0.000001
act_input = 'tanh'
act_output = 'tanh'
epochs = 50
batch_size = 5

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