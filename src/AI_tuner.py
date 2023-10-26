from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import itertools
import random
import datetime
import pandas as pd
import numpy as np
from src.AI_fun import *

# Disable all warnings
warnings.filterwarnings('ignore')

'''
                                       ---------- Hyperparameter Tuner -----------

This tool will try combinations of indicators and then try combinations of hyperparameters to find the model with the 
best accuracy in the prediction of the price movement. 
'''


def tuner(dataset_name=dataset_name):
    dataset_name = 't_' + dataset_name
    resample_size = f'{str(resample_co)}s'
    idx = (time_steps + future_points * sequences) * validations
    w0 = 28 * future_points
    limi = w0 + idx
    last_reset = datetime.now()
    limit_len = limi * resample_co
    since = round_up(limit_len / 3600,
                     2)  # Since must be expressed in hours if limit len is expressed in seconds  we divide by 3600.
    hyper_list = ['layers_list', 'epochs_list', 'batch_size_list', 'dropout_rate_list', 'learning_rate_list',
                  'activation_output_list']

    Import_AI(name=dataset_name, cryptos=crypto, sample_freq=interval, since=since, future_points=future_points,
              resample_size=resample_size)
    print(f'Retrieving dataset: {dataset_name}')
    dataset = pd.read_csv(dataset_name, index_col=['datetime'], parse_dates=True).fillna(method='ffill')
    acc_mean = 0
    loss_mean = 0

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
        input_data, target_data = prepare_input_target_data(dataset, feat_num)

        # Create separate scalers for each category of features
        open_scaler, high_scaler, low_scaler, close_scaler, volume_scaler, so_scaler, pvo_scaler, ema_scaler, rsi_scaler, srsi_scaler, cci_scaler, psar_scaler, vwap_scaler, target_scaler = create_separate_scalers()

        # Standardize and normalize the price data
        open_data = standardize_and_normalize(input_data, scaler=open_scaler, data_axis=0)
        high_data = standardize_and_normalize(input_data, scaler=high_scaler, data_axis=1)
        low_data = standardize_and_normalize(input_data, scaler=low_scaler, data_axis=2)
        close_data = standardize_and_normalize(input_data, scaler=close_scaler, data_axis=3)

        # Standardize and normalize EMA, PSAR and VWAP
        ema_data = standardize_and_normalize(input_data, ema_scaler, data_axis=3, p_len=4)
        psar_data = standardize_and_normalize(input_data, psar_scaler, data_axis=7, p_len=4)
        vwap_data = standardize_and_normalize(input_data, vwap_scaler, data_axis=8, p_len=4)

        # Normalize SO, WI, PVO, SRSI & CCI
        so_data = normalize(input_data, so_scaler, data_axis=1, p_len=4)
        pvo_data = normalize(input_data, pvo_scaler, data_axis=2, p_len=4)
        rsi_data = normalize(input_data, rsi_scaler, data_axis=4, p_len=4)
        srsi_data = normalize(input_data, srsi_scaler, data_axis=5, p_len=4)
        cci_data = normalize(input_data, cci_scaler, data_axis=6, p_len=4)

        # Standardize the volume data
        volume_data = normalize(input_data, volume_scaler, data_axis=4)

        # List of arrays to choose from
        arr_dic = {'so': so_data, 'pvo': pvo_data, 'ema': ema_data, 'rsi': rsi_data,
                   'srsi': srsi_data, 'cci': cci_data, 'psar': psar_data, 'vwap': vwap_data}
        array = [arr_dic[arr_list[i]] for i in range(len(arr_list))]
        combinations = []

        for r in range(0, len(array) + 1):
            combinations.extend(list(map(list, itertools.combinations(array, r))))

        if hyper_mode == 0:
            random.shuffle(combinations)
            comb_index = comb_loops * -1
        elif hyper_mode == 1:
            comb_index = -1
        else:
            random.shuffle(combinations)
            comb_index = 0

        best_score = 0
        best_combination = None

        for selected_arrays in combinations[comb_index:]:
            input_data = np.concatenate((open_data, high_data, low_data, close_data, volume_data, *selected_arrays),
                                        axis=2)
            combination_indexes = []
            for h in selected_arrays:
                index = np.where((array == h).all(axis=1))[0][0]
                combination_indexes.append(index)
            comb_list = [arr_list[combination_indexes[j]] for j in range(len(combination_indexes))]
            print(f'Testing Combination: {comb_list}')
            acc_mean = 0
            loss_mean = 0

            if hyper_mode != 3:
                for y in range(hyper_loops):

                    for u in hyper_list:
                        random.shuffle(hyper_dic[u])

                    layers = hyper_dic['layers_list'][0]
                    epochs = hyper_dic['epochs_list'][0]
                    batch_size = hyper_dic['batch_size_list'][0]
                    dropout = hyper_dic['dropout_rate_list'][0]
                    learning_rate = hyper_dic['learning_rate_list'][0]
                    act_input = hyper_dic['activation_input_list'][0]
                    act_output = hyper_dic['activation_output_list'][0]

                    print(
                        f'Layers: {layers} Epochs: {epochs} Batch Size: {batch_size} Dropout Rate: {dropout} Learning Rate: {learning_rate} Input Act. Func: {act_input} Output Act. Func: {act_output}')
                    acc_mean = 0
                    loss_mean = 0
                    for z in range(validations):
                        ix = int(len(input_data) / validations) * z
                        fx = ix + int(len(input_data) / validations)
                        input_d = input_data[ix:fx, :, :]
                        target_d = target_data[ix:fx]

                        # Create Model
                        model = create_model(input_d, num_lstm_units=layers, dropout_rate=dropout,
                                             learning_rate=learning_rate,
                                             activation_input=act_input, activation_output=act_output)
                        # Set train/test
                        input_train, input_test, target_train, target_test = train_test_split(
                            input_d, target_d, test_size=test_size, shuffle=False)
                        # Fit the model
                        model.fit(input_train, target_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
                        accuracy, loss = evaluate(model, input_test, target_test)
                        acc_mean += accuracy / validations
                        loss_mean += loss / validations
                    print(f'Average Accuracy: {acc_mean}')
                    print(f'Average Loss: {loss_mean}')
                    if acc_mean > best_score:
                        best_score, best_combination, best_model, best_combination_indexes = accuracy, selected_arrays, model, combination_indexes
                        print('Best accuracy so far!')
            else:
                print(
                    f'Layers: {layers} Epochs: {epochs} Batch Size: {batch_size} Dropout Rate: {dropout} Learning Rate: {learning_rate} Input Act. Func: {act_input} Output Act. Func: {act_output}')
                acc_mean = 0
                loss_mean = 0
                for z in range(validations):
                    ix = int(len(input_data) / validations) * z
                    fx = ix + int(len(input_data) / validations)
                    input_d = input_data[ix:fx, :, :]
                    target_d = target_data[ix:fx]

                    # Create Model
                    model = create_model(input_d, num_lstm_units=layers, dropout_rate=dropout,
                                         learning_rate=learning_rate,
                                         activation_input=act_input, activation_output=act_output)
                    # Set train/test
                    input_train, input_test, target_train, target_test = train_test_split(
                        input_d, target_d, test_size=test_size, shuffle=False)
                    # Fit the model
                    model.fit(input_train, target_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
                    accuracy, loss = evaluate(model, input_test, target_test)
                    acc_mean += accuracy / validations
                    loss_mean += loss / validations
                print(f'Average Accuracy: {acc_mean}')
                print(f'Average Loss: {loss_mean}')
                if acc_mean > best_score:
                    best_score, best_combination, best_model, best_combination_indexes = accuracy, selected_arrays, model, combination_indexes
                    print('Best accuracy so far!')

        print(f"Best combination of Indicators: {comb_list}")
        # print("Best score of the best combination:", best_score)
        print(
            f'Best Hyperparameters: Layers:{layers} Dropout Rate:{dropout} Learning Rate: {learning_rate} Input Act. Fun.: {act_input} Output Act. Fun.: {act_output} Epochs: {epochs} Batch Size:{batch_size}')
        elapsed_time = (datetime.now() - last_reset).total_seconds()
        # Check if it's time to reset the variable
        print(f'Modeling duration: {datetime.now() - start_t}')

    else:
        print('Dataset is not long enough for modeling. Exiting...')
