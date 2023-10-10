import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from datetime import datetime, timedelta
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import save_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
import itertools
import joblib
import time
import random
import os
import logging
import math
from AI_Func import *


'''
                                       ---------- Hyperparameter Tuner -----------
                                       
This tool will try combinations of indicators and then try combinations of hyperparameters to find the model with the 
best accuracy in the prediction of the price movement. 
'''

dataset_name = 't_' + dataset_name
idx = time_steps + future_points * sequences
w0 = 28 * future_points
limi = w0 + idx
the_best_score = 0
last_reset = datetime.now()
limit_len = limi * conv_to_interval
since = round_up(limit_len/3600, 2) # Since must be expressed in hours if limit len is expressed in seconds  we divide by 3600.
Import_AI(name=dataset_name, cryptos=crypto, sample_freq=interval, since=since, future_points=future_points, resample_size=resample_size)

while True:

    # ETL CSV
    while True:
        try:
            print(f'Retrieving dataset: {dataset_name}')
            dataset = pd.read_csv(dataset_name, index_col=['datetime'], parse_dates=True).fillna(method='ffill')
        except:
            time.sleep(60)
            continue
        break

    if len(dataset) > limi:
        print('Starting modeling...')
        start_t = datetime.now()
        print(f'Current Datetime:{start_t}')

        dataset['Return'] = dataset.close.pct_change(future_points).shift(-future_points)
        dataset['Return'] = np.where(dataset['Return'] > 0, 1, 0)

        feat_num = len(dataset.columns)  # The dataset contains x features columns and 1 column as target ('Return')
        dataset = dataset[-idx:]
        dataset.dropna(inplace=True)

        # Separate columns based on types
        price_columns = ['open', 'close', 'high', 'low']
        volume_column = ['volume']
        indicator_columns = ['so', 'pvo', 'ema', 'rsi', 'srsi', 'cci', 'psar', 'vwap']

        # Prepare the data
        input_data = []
        target_data = []

        for i in range(0, len(dataset) - time_steps + 1, future_points):
            input_data.append(dataset.iloc[i:i + time_steps, :feat_num].values)
            target_data.append(dataset.iloc[i + time_steps - 1, dataset.columns.get_loc('Return')])

        input_data = np.array(input_data)
        target_data = np.array(target_data)

        # Create separate scalers for each category of features

        open_scaler = StandardScaler()
        high_scaler = StandardScaler()
        low_scaler = StandardScaler()
        close_scaler = StandardScaler()
        volume_scaler = StandardScaler()
        so_scaler = MinMaxScaler()
        pvo_scaler = StandardScaler()
        ema_scaler = StandardScaler()
        rsi_scaler = MinMaxScaler()
        srsi_scaler = MinMaxScaler()
        cci_scaler = MinMaxScaler()
        psar_scaler = StandardScaler()
        vwap_scaler = StandardScaler()
        target_scaler = StandardScaler()


        # Standardize and normalize the price data
        try:
            open_data = input_data[:, :, 0]
            # save_data = input_data.reshape(-1, input_data.shape[-1])
            # np.savetxt('correct_input_data.csv', save_data, delimiter=',')
        except:
            print('Error!!!')
            print(input_data)
            save_data = input_data.reshape(-1, input_data.shape[-1])
            np.savetxt('wrong_input_data.csv', save_data, delimiter=',')
            time.sleep(30)
            continue
        open_data = (open_data - np.mean(open_data, axis=0)) / np.std(open_data, axis=0)
        open_data = open_scaler.fit_transform(open_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
        high_data = input_data[:, :, 1]
        high_data = (high_data - np.mean(high_data, axis=0)) / np.std(high_data, axis=0)
        high_data = high_scaler.fit_transform(high_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
        low_data = input_data[:, :, 2]
        low_data = (low_data - np.mean(low_data, axis=0)) / np.std(low_data, axis=0)
        low_data = low_scaler.fit_transform(low_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
        close_data = input_data[:, :, 3]
        close_data = (close_data - np.mean(close_data, axis=0)) / np.std(close_data, axis=0)
        close_data = close_scaler.fit_transform(close_data.reshape(-1, 1)).reshape(-1, time_steps, 1)

        # Standardize and normalize EMA and PSAR
        ema_data = input_data[:, :, len(price_columns) + 3]
        ema_data = (ema_data - np.mean(ema_data, axis=0)) / np.std(ema_data, axis=0)
        ema_data = ema_scaler.fit_transform(ema_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
        psar_data = input_data[:, :, len(price_columns) + 7]
        psar_data = (psar_data - np.mean(psar_data, axis=0)) / np.std(psar_data, axis=0)
        psar_data = psar_scaler.fit_transform(psar_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
        vwap_data = input_data[:, :, len(price_columns) + 8]
        vwap_data = (vwap_data - np.mean(vwap_data, axis=0)) / np.std(vwap_data, axis=0)
        vwap_data = vwap_scaler.fit_transform(vwap_data.reshape(-1, 1)).reshape(-1, time_steps, 1)

        # Normalize SO, WI, PVO, SRSI & CCI
        so_data = input_data[:, :, len(price_columns) + 1]
        so_data = so_scaler.fit_transform(so_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
        pvo_data = input_data[:, :, len(price_columns) + 2]
        pvo_data = pvo_scaler.fit_transform(pvo_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
        rsi_data = input_data[:, :, len(price_columns) + 4]
        rsi_data = rsi_scaler.fit_transform(rsi_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
        srsi_data = input_data[:, :, len(price_columns) + 5]
        srsi_data = srsi_scaler.fit_transform(srsi_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
        cci_data = input_data[:, :, len(price_columns) + 6]
        cci_data = cci_scaler.fit_transform(cci_data.reshape(-1, 1)).reshape(-1, time_steps, 1)

        # Standardize the volume data
        volume_data = input_data[:, :, len(price_columns)]
        volume_data = volume_scaler.fit_transform(volume_data.reshape(-1, 1)).reshape(-1, time_steps, 1)

        # Normalize the output data
        #target_data = target_scaler.fit_transform(target_data.reshape(-1, 1)).reshape(target_data.shape)


        # Define the LSTM model
        def create_model(num_lstm_units, dropout_rate, learning_rate, activation_input, activation_output):
            model = Sequential()
            model.add(LSTM(num_lstm_units, input_shape=(time_steps, input_data.shape[2]), activation=activation_input,
                           dropout=dropout_rate))
            model.add(Dense(1, activation=activation_output))
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            return model


        # List of arrays to choose from
        arr_dic = {'so_data': so_data, 'pvo_data': pvo_data, 'ema_data': ema_data, 'rsi_data': rsi_data, 'srsi_data': srsi_data, 'cci_data': cci_data, 'psar_data': psar_data, 'vwap_data': vwap_data}
        array = [arr_dic[arr_list[i]] for i in range(len(arr_list))]

        combinations = []
        for r in range(0, len(array) + 1):
            combinations.extend(list(map(list, itertools.combinations(array, r))))

        best_score = 0
        best_combination = None

        epochs_list = [5, 10, 30, 50, 60, 70, 90, 120, 240]
        batch_size_list = [1, 5, 10, 15, 25, 50]
        num_lstm_units_list = [1, 2, 4, 6, 8, 10, 32, 64, 128, 256]
        dropout_rate_list = [0.1, 0.2, 0.3, 0.5]
        learning_rate_list = [0.001, 0.0001, 0.00001, 0.000001]
        activation_input_list = ['tanh', 'sigmoid']
        activation_output_list = ['tanh', 'sigmoid']
        random.shuffle(epochs_list)
        random.shuffle(batch_size_list)
        random.shuffle(num_lstm_units_list)
        random.shuffle(dropout_rate_list)
        random.shuffle(learning_rate_list)
        random.shuffle(activation_input_list)
        random.shuffle(activation_output_list)

        if hyper_mode != 2:
            layers = num_lstm_units_list[0]
            dropout = dropout_rate_list[0]
            learning_rate = learning_rate_list[0]
            act_input = activation_input_list[0]
            act_output = activation_output_list[0]
            epochs = epochs_list[0]
            batch_size = batch_size_list[0]
            if hyper_mode == 1:
                comb_index = -1
        else:
            comb_index = 0
            random.shuffle(combinations)
        params = [layers, dropout, learning_rate, act_input, act_output, epochs, batch_size]

        for selected_arrays in combinations[comb_index:]:
            input_data = np.concatenate((open_data, high_data, low_data, close_data, volume_data, *selected_arrays),
                                        axis=2)
            combination_indexes = []
            for h in selected_arrays:
                index = np.where((array == h).all(axis=1))[0][0]
                combination_indexes.append(index)
            comb_list = [arr_list[combination_indexes[j]] for j in range(len(combination_indexes))]
            logger.info(f'Testing Combination:{comb_list}')

            # Create and fit the LSTM model with the specified hyperparameters
            model = create_model(num_lstm_units=layers, dropout_rate=dropout, learning_rate=learning_rate,
                                 activation_input=act_input,
                                 activation_output=act_output)

            input_train, input_test, target_train, target_test = train_test_split(
                input_data, target_data, test_size=0.25, shuffle=False)

            # print(input_train.shape, target_train.shape, input_test.shape, target_test.shape)

            # Fit the model
            model.fit(input_train, target_train, epochs=epochs, batch_size=batch_size, verbose=0)

            # Evaluate the model on the test set
            loss, accuracy = model.evaluate(input_test, target_test, verbose=0)
            predictions = model.predict(input_test)
            predictions = np.where(predictions >= 0.5, 1, 0)  # Convert probabilities to binary predictions
            predictions = predictions.flatten()
            accuracy = np.mean(predictions == target_test)  # Calculate accuracy
            logger.info(f'Accuracy: {accuracy:.2f}')
            logger.info(f'Score:{loss}')

            # Check if the current combination is better than the previous best
            if accuracy > best_score:
                best_score = accuracy
                best_combination = selected_arrays
                best_model = model
                best_combination_indexes = combination_indexes

        logger.info(f"Best combination of arrays:{best_combination_indexes}")
        # logger.info("Best score of the best combination:", best_score)
        logger.info(
            f'Layers:{layers} Dropout Rate:{dropout} Learning Rate: {learning_rate} Input Act. Fun.: {act_input} Output Act. Fun.: {act_output} Epochs: {epochs} Batch Size:{batch_size}')

        scalers = {
            'open_scaler': open_scaler,
            'high_scaler': high_scaler,
            'low_scaler': low_scaler,
            'close_scaler': close_scaler,
            'volume_scaler': volume_scaler,
            'target_scaler': target_scaler,
            'so_scaler': so_scaler,
            'pvo_scaler': pvo_scaler,
            'ema_scaler': ema_scaler,
            'rsi_scaler': rsi_scaler,
            'srsi_scaler': srsi_scaler,
            'cci_scaler': cci_scaler,
            'psar_scaler': psar_scaler,
            'vwap_scaler': vwap_scaler,
            'target_scaler': target_scaler,
        }
        elapsed_time = (datetime.now() - last_reset).total_seconds()

        # Check if it's time to reset the variable
        if elapsed_time > 21600:
            #the_best_score = float('inf')
            the_best_score = 0
            last_reset = datetime.now()

        if best_score > the_best_score:
            logger.info('Best score so far')
            # Save model and Scalers
            the_best_score = best_score
            joblib.dump(params, f'{loc_folder}best_params.joblib')

        print(f'Modeling duration: {datetime.now() - start_t}')
    else:
        print('Dataset is not long enough yet for modeling')
        time.sleep(60)