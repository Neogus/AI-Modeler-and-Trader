from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import itertools
import random
from AI_Func import *
# Disable all warnings
warnings.filterwarnings('ignore')

'''
                                       ---------- Hyperparameter Tuner -----------

This tool will try combinations of indicators and then try combinations of hyperparameters to find the model with the 
best accuracy in the prediction of the price movement. 
'''

dataset_name = 't_' + dataset_name
resample_size = f'{str(resample_co)}s'
idx = (time_steps + future_points * sequences) * validations
w0 = 28 * future_points
limi = w0 + idx
last_reset = datetime.now()
limit_len = limi * resample_co
since = round_up(limit_len / 3600,
                 2)  # Since must be expressed in hours if limit len is expressed in seconds  we divide by 3600.
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
    open_data = input_data[:, :, 0]
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
                random.shuffle(hyper_dic['layers_list'])
                random.shuffle(hyper_dic['epochs_list'])
                random.shuffle(hyper_dic['batch_size_list'])
                random.shuffle(hyper_dic['dropout_rate_list'])
                random.shuffle(hyper_dic['learning_rate_list'])
                random.shuffle(hyper_dic['activation_input_list'])
                random.shuffle(hyper_dic['activation_output_list'])
                layers = hyper_dic['layers_list'][0]
                epochs = hyper_dic['epochs_list'][0]
                batch_size = hyper_dic['batch_size_list'][0]
                dropout = hyper_dic['dropout_rate_list'][0]
                learning_rate = hyper_dic['learning_rate_list'][0]
                act_input = hyper_dic['activation_input_list'][0]
                act_output = hyper_dic['activation_output_list'][0]

                print(f'Layers: {layers} Epochs: {epochs} Batch Size: {batch_size} Dropout Rate: {dropout} Learning Rate: {learning_rate} Input Act. Func: {act_input} Output Act. Func: {act_output}')
                acc_mean = 0
                loss_mean = 0
                for z in range(validations):
                    ix = int(len(input_data)/validations) * z
                    fx = ix + int(len(input_data)/validations)
                    input_d = input_data[ix:fx, :, :]
                    target_d = target_data[ix:fx]

                    # Create Model
                    model = create_model(input_d, num_lstm_units=layers, dropout_rate=dropout, learning_rate=learning_rate,
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
            print(f'Layers: {layers} Epochs: {epochs} Batch Size: {batch_size} Dropout Rate: {dropout} Learning Rate: {learning_rate} Input Act. Func: {act_input} Output Act. Func: {act_output}')
            acc_mean = 0
            loss_mean = 0
            for z in range(validations):
                ix = int(len(input_data) / validations) * z
                fx = ix + int(len(input_data) / validations)
                input_d = input_data[ix:fx, :, :]
                target_d = target_data[ix:fx]

                # Create Model
                model = create_model(input_d, num_lstm_units=layers, dropout_rate=dropout, learning_rate=learning_rate,
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
    print(f'Best Hyperparameters: Layers:{layers} Dropout Rate:{dropout} Learning Rate: {learning_rate} Input Act. Fun.: {act_input} Output Act. Fun.: {act_output} Epochs: {epochs} Batch Size:{batch_size}')
    elapsed_time = (datetime.now() - last_reset).total_seconds()
    # Check if it's time to reset the variable
    print(f'Modeling duration: {datetime.now() - start_t}')

else:
    print('Dataset is not long enough for modeling. Exiting...')
