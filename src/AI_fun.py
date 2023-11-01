from datetime import datetime, timedelta
import math
import ccxt
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from ta.momentum import RSIIndicator, StochRSIIndicator, StochasticOscillator
from ta.trend import CCIIndicator, EMAIndicator
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import joblib
import websocket
import json
import random
import threading
from src.AI_config import *
from src.API_config import *  # Fill-up the API_config file with your keys

# Disable all warnings
warnings.filterwarnings('ignore')


hyper_dic = {'epochs_list': [5, 10, 30, 50, 60, 70, 90, 120, 240],
        'batch_size_list': [1, 5, 10, 15, 25, 50],
        'layers_list': [1, 2, 4, 6, 8, 10, 32, 64, 128, 256],
        'dropout_rate_list': [0.1, 0.2, 0.3, 0.5],
        'learning_rate_list': [0.001, 0.0001, 0.00001, 0.000001],
        'activation_input_list': ['tanh', 'sigmoid'],
        'activation_output_list': ['tanh', 'sigmoid']}

def round_down(num, dec):
    num = math.floor(num * 10 ** dec) / 10 ** dec
    return num

def round_up(num, dec):
    num = math.ceil(num * 10 ** dec) / 10 ** dec
    return num

def get_psar(df, iaf=0.02, maxaf=0.2):
    length = len(df)
    high = df['high']
    low = df['low']
    df['psar'] = df['close'].copy()
    bull = True
    af = iaf
    hp = high.iloc[0]
    lp = low.iloc[0]

    for i in range(2, length):
        hp_lp = hp if bull == True else lp
        df.psar.iloc[i] = df.psar.iloc[i - 1] + af * (hp_lp - df.psar.iloc[i - 1])

        reverse = False

        if bull:
            if low.iloc[i] < df.psar.iloc[i]:
                bull = False
                reverse = True
                df.psar.iloc[i] = hp
                lp = low.iloc[i]
                af = iaf
        else:
            if high.iloc[i] > df.psar.iloc[i]:
                bull = True
                reverse = True
                df.psar.iloc[i] = lp
                hp = high.iloc[i]
                af = iaf

        if not reverse:
            if bull:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + iaf, maxaf)
                if low.iloc[i - 1] < df.psar.iloc[i]:
                    df.psar.iloc[i] = low[i - 1]
                if low.iloc[i - 2] < df.psar.iloc[i]:
                    df.psar.iloc[i] = low.iloc[i - 2]
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + iaf, maxaf)
                if high.iloc[i - 1] > df.psar.iloc[i]:
                    df.psar.iloc[i] = high.iloc[i - 1]
                if high.iloc[i - 2] > df.psar.iloc[i]:
                    df.psar.iloc[i] = high.iloc[i - 2]
    return df.psar

def standardize_and_normalize(input_data, scaler, data_axis, p_len=0, epsilon=1e-10):
    extracted_data = input_data[:, :, p_len + data_axis]
    extracted_data = (extracted_data - np.mean(extracted_data, axis=0)) / np.std(extracted_data + epsilon, axis=0)
    extracted_data = scaler.fit_transform(extracted_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    return extracted_data

def normalize(input_data, scaler, data_axis, p_len=0):
    extracted_data = input_data[:, :, p_len + data_axis]
    extracted_data = scaler.fit_transform(extracted_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    return extracted_data

def create_separate_scalers():
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
    return open_scaler,high_scaler,low_scaler,close_scaler,volume_scaler,so_scaler,pvo_scaler,ema_scaler,rsi_scaler,srsi_scaler,cci_scaler,psar_scaler,vwap_scaler,target_scaler

def prepare_input_target_data(dataset, feat_num):
    input_data = []
    target_data = []
    for i in range(0, len(dataset) - time_steps + 1, future_points):
        input_data.append(dataset.iloc[i:i + time_steps, :feat_num].values)
        target_data.append(dataset.iloc[i + time_steps - 1, dataset.columns.get_loc('Return')])
    input_data = np.array(input_data)
    target_data = np.array(target_data)
    return input_data,target_data


def create_model(input_data, num_lstm_units, dropout_rate, learning_rate, activation_input, activation_output):
    model = Sequential()
    model.add(LSTM(num_lstm_units, input_shape=(time_steps, input_data.shape[2]), activation=activation_input,
                   dropout=dropout_rate))
    model.add(Dense(1, activation=activation_output))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate(model, input_test, target_test):
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(input_test, target_test, verbose=verbose)
    predictions = model.predict(input_test)
    predictions = np.where(predictions >= 0.5, 1, 0)  # Convert probabilities to binary predictions
    predictions = predictions.flatten()
    accuracy = np.mean(predictions == target_test)  # Calculate accuracy
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Score: {loss}')
    return accuracy, loss

def fetch_data(exchange='binance', cryptos=['BTC/USDT'], sample_freq='1m', since_hours=48, page_limit=1000, max_retries = 3):

    since = (datetime.today() - timedelta(hours=since_hours) - timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%S')
    print('Begin download...')

    for market_symbol in cryptos:

        # Select exchange
        exchange = getattr(ccxt, exchange)({'enableRateLimit': True, })

        # Convert since from string to milliseconds
        since = exchange.parse8601(since)

        # Preload all markets from the exchange
        exchange.load_markets()

        # Define page_limit in milliseconds
        earliest_timestamp = exchange.milliseconds()
        ms_timeframe = exchange.parse_timeframe(sample_freq) * 1000 # exchange.parse_timeframe parses to the equivalent in seconds of the timeframe we use
        t_delta = page_limit * ms_timeframe
        all_ohlcv = []
        num_retries = 0
        fetch_since = earliest_timestamp - t_delta

        while True:

            try:
                num_retries += 1
                ohlcv = exchange.fetch_ohlcv(market_symbol, sample_freq, fetch_since, page_limit)

            except Exception as e:
                print(str(e))
                time.sleep(5)
                print('Retrying...')
                if num_retries > max_retries:
                    print('Could not connect with exchange. Exiting...')
                    exit()
                continue

            earliest_timestamp = ohlcv[0][0]
            all_ohlcv = ohlcv + all_ohlcv
            # if we have reached the checkpoint
            if fetch_since < since:
                break
            fetch_since = earliest_timestamp - t_delta

        ohlcv = exchange.filter_by_since_limit(all_ohlcv, since, None, key=0)
        df = pd.DataFrame(ohlcv)

        if market_symbol == cryptos[0]:

            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], unit='ms')
            df.rename(columns={0: 'datetime', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
            df = df.set_index('datetime')
            dfx = df.copy()

        else:

            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], unit='ms')
            df.rename(columns={0: 'datetime', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
            df = df.set_index('datetime')
            dfx = pd.merge(dfx, df, on=['datetime'])

    dfx = dfx.loc[:, ~dfx.columns.duplicated()]
    dfx = dfx[~dfx.index.duplicated(keep='first')]

    print(f'Finished')
    return dfx

def import_csv(loc_folder, filename):

    read_file = f'{loc_folder}/{filename}'
    df = pd.read_csv(read_file, index_col='Datetime', parse_dates=True)
    return df

def calc_ti(df, w, w0, w1, w2, w3,resample_size):

    df = df.resample(resample_size).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).fillna(
        method='ffill')

    # Calculate Stochastic Oscillator
    stochastic = StochasticOscillator(df["high"], df["low"], df["close"], window=w0)
    df["so"] = stochastic.stoch()

    # Calculate Percentage Volume Oscillator (PVO) manually
    ema_short = df["volume"].rolling(window=w1, min_periods=w).mean()
    ema_long = df["volume"].rolling(window=w0, min_periods=w).mean()
    pvo = (ema_short - ema_long) / ema_long
    df["pvo"] = pvo

    # Calculate Exponential Moving Average (EMA)
    ema = EMAIndicator(df["close"], window=w2)
    df['ema'] = (df['close'] - (ema.ema_indicator())) / df['close']

    # Calculate Stochastic RSI
    rsi = RSIIndicator(df["close"], window=w1)
    df['rsi'] = rsi.rsi()
    stoch_rsi = StochRSIIndicator(rsi.rsi(), window=w1, smooth1=w2, smooth2=w3)
    df["srsi"] = stoch_rsi.stochrsi()

    # Calculate Commodity Channel Index (CCI)
    cci = CCIIndicator(df["high"], df["low"], df["close"], window=w0)
    df["cci"] = cci.cci()

    # Calculate Parabolic SAR (PSAR)
    df['psar'] = (df['close'] - (get_psar(df, iaf=0.0002, maxaf=0.2))) / df['close']

    # Calculate the VWAP
    df['vwap'] = (df['close'] - ((df['volume'] * df['close']).cumsum() / df['volume'].cumsum())) / df['close']

    return df

def Import_AI(name='candles.csv', cryptos=['BTC/TUSD'], sample_freq='1s', since=48, future_points = 9, resample_size='1s'):
    print(f'{cryptos[0]}')
    w0 = future_points * 28
    w1 = int(w0 / 2)
    w2 = int(w1 / 2)
    w3 = int(w2 / 2)
    dfx = fetch_data(exchange='binance', cryptos=cryptos, sample_freq=sample_freq, since_hours=since, page_limit=1000)
    dfx = calc_ti(dfx, future_points, w0, w1, w2, w3, resample_size)
    dfx.to_csv(name)
    print(f'{cryptos[0]} Historical Length: {len(dfx)}')

def append_to_col(dft, col, val):
    idx = dft[col].last_valid_index()
    dft.loc[idx + 1 if idx is not None else 0, col] = val
    return dft

def ret_threshold(df2):
    best_acc = 0.5
    best_ret = 0
    best_rate = 0
    best_thres = 0
    best_t = 0

    for x in range(60):

        aux = df2.copy()
        ret_t = x * 0.00005

        # Buy and sell 2 minutes after
        aux['Trade'] = aux['Current Price'].pct_change().shift(-1)
        aux.loc[(aux['Predicted Return %'] < -ret_t) & (aux['Trade'] < 0), 'Trade2'] = -aux['Trade']
        aux.loc[(aux['Predicted Return %'] < -ret_t) & (aux['Trade'] > 0), 'Trade2'] = -aux['Trade']
        aux.loc[(aux['Predicted Return %'] > ret_t) & (aux['Trade'] > 0), 'Trade2'] = aux['Trade']
        aux.loc[(aux['Predicted Return %'] > ret_t) & (aux['Trade'] < 0), 'Trade2'] = aux['Trade']
        aux.dropna(inplace=True)
        ret = aux['Trade2']



        t_num = len(ret)
        if t_num < 1:
            continue
        elif t_num < 2:
            t_ret = ret.sum()
            # print(f'Ret sum:{t_ret}')
        else:
            t_ret = (1 + ret).cumprod()
            # print(t_ret)
            #print(t_ret.max(), t_ret.idxmax())
            t_ret = t_ret[-1] - 1
            # print(f'Ret Prod:{t_ret}')

        t_rate = t_ret / t_num
        acc = (ret > 0).sum() / t_num
        #print(f'Thresh:{ret_t:.6f} Ret:{t_ret:.6f} Acc:{acc:.2f} T.Num:{t_num} T.Rate:{t_rate:.6f}')

        if t_ret > best_ret:
            best_acc = acc
            best_ret = t_ret
            best_rate = t_rate
            best_thres = ret_t
            best_t = t_num
    if best_ret == 0:
        'Not enough predictions to calculate Return Threshold, using default (0%)'
        return 1
    else:
        print(f'Prediction Analysis Results\n'
              f'Best Threshold:{best_thres:.6f}. Acc={best_acc * 100:.2f}% Ret={best_ret * 100:.4f}% T.Num={best_t} T.Rate={best_rate * 100:.4f}%')
    return round(best_thres, 5)

def log_status(dfs, cdf):
    logger.info(f'Status:\n{dfs} \nSignals: {cdf}')

def place_order(ticker, asset, asset_m, side, e_type, price_point, bot, client):
    exit_code = 0
    order_loop = 0
    order_name = f'{side}'
    while True:
        try:
            exit_loop = 0
            info = client.get_margin_account()
            logger.info('Info Retrieved')
            for x in range(len(info['userAssets'])):
                if info['userAssets'][x]['asset'] == asset:
                    amount = float(info['userAssets'][x][asset_m]) + float(info['userAssets'][x]['interest'])
                    break

            logger.info('Info Loop Ok')
            ticker_price = client.get_orderbook_ticker(symbol=ticker)
            btc_price = client.get_orderbook_ticker(symbol='BTCUSDT')
            logger.info('Info Ticker Retrieved')
            balance = float(btc_price['askPrice']) * float(info['totalNetAssetOfBtc'])
            logger.info(f'Balance:{balance}')
            bid_price = float(ticker_price['bidPrice'])
            ask_price = float(ticker_price['askPrice'])
            logger.info('Info Ticker Retrieved')
            if e_type == 'AUTO_REPAY':
                price = round_down((bid_price + ask_price) / 2, 2)
                amount = round_up(amount, asset_precision)
                order_name = 'pago'
            elif e_type == 'NO_SIDE_EFFECT' and side == 'BUY':
                price = round_down((bid_price + ask_price) / 2, 2)
                amount = round_down(amount / price, asset_precision)
                order_name = 'compra'
            elif e_type == 'NO_SIDE_EFFECT' and side == 'SELL':
                price = round_up((bid_price + ask_price) / 2, 2)
                amount = round_down(amount, asset_precision)
                order_name = 'venta'
            elif e_type == 'MARGIN_BUY':
                price = round_up((bid_price + ask_price) / 2, 2)
                amount = round_down(amount / price, asset_precision)
                order_name = 'prestamo'
            logger.info('Rounded Finished')
            logger.info(f'{price},{amount},{ticker},{side},{asset},{asset_m},{e_type}')
            if price * amount > 10:
                #order = client.create_margin_order(symbol=ticker, side=side, type="LIMIT", timeInForce='GTC',
                #                                   quantity=amount, price=price, sideEffectType=e_type)
                order = client.create_margin_order(symbol=ticker, side=side, type="MARKET",
                                                   quantity=amount, sideEffectType=e_type)
                logger.info('Order Placed Ok')
            else:
                message = f'Balance insuficiente para efectuar la orden de {order_name}.'
                logger.info(message)
                # telegram_send.send(messages=[message])
                bot.send_message(chat_id, message)
                exit_code = 1
                return exit_code, price_point
        except Exception as e:
            print("An error occurred:", str(e))
            time.sleep(5)
            order_loop += 1
            if order_loop > 3:
                logger.info(f'No se ha podido colocar la orden de {order_name} debido a un error de conexión.')
                exit_code = 2
                return exit_code, price_point
            else:
                continue

        check_loop = 0
        error_loop = 0
        time.sleep(5)
        order = client.get_margin_order(symbol=ticker, orderId=order['orderId'])
        while order['status'] != 'FILLED':
            try:
                if check_loop > 3 and order['status'] != 'PARTIALLY_FILLED':
                    logger.info(f'Se ha acabado el tiempo para completar la orden de {order_name}.')
                    client.cancel_margin_order(symbol=ticker, orderId=order['orderId'])
                    while order['status'] != 'CANCELED':
                        logger.info(f'Esperando cancelación')
                        time.sleep(5)
                        order = client.get_margin_order(symbol=ticker, orderId=order['orderId'])
                    logger.info(f'La orden fue cancelada.')
                    exit_loop = 1
                    break
                else:

                    logger.info(f'La orden no fue completada todavía...')
                    time.sleep(5)
                    check_loop += 1
                    order = client.get_margin_order(symbol=ticker, orderId=order['orderId'])
                    if order['status'] != 'PARTIALLY_FILLED':
                        logger.info(
                            f'La orden fue completada parcialmente, no puede ser cancelada hasta que termine de completarse.')

                    continue
            except:
                error_loop += 1
                if error_loop > 3:
                    message = f'No se ha podido verificar la ejecución de la orden de {order_name} debido a un error de conexión. Se requiere intervención manual.'
                    logger.info(message)
                    # telegram_send.send(messages=[message])
                    bot.send_message(chat_id, message)
                    exit_code = 3  # Check failed
                    return exit_code
                else:
                    time.sleep(5)
                    continue
        if exit_loop == 0:
            break

    if e_type == 'AUTO_REPAY':
        equivalent = price_point * amount
        t_return = (price / price_point - 1) * -100
        message = f'He pagado el préstamo de {amount} {ticker[:-4]} (${equivalent:.2f}) a un precio de ${price:.2f}, obteniendo un retorno de {t_return:.2f}%.'
    elif e_type == 'NO_SIDE_EFFECT' and side == 'BUY':
        equivalent = amount * price
        message = f'He comprado {amount} {ticker[:-4]} (${equivalent:.2f}) a un precio de ${price} \n' \
                  f'Balance actual = ${balance:.2f}'
    elif e_type == 'NO_SIDE_EFFECT' and side == 'SELL':
        equivalent = amount * price
        t_return = (price / price_point - 1) * 100
        message = f'He vendido {amount} {ticker[:-4]} (${equivalent:.2f}) a un precio de ${price}, obteniendo un retorno de {t_return:.2f}%. \n' \
                  f'Balance actual = ${balance:.2f}'
    elif e_type == 'MARGIN_BUY':
        equivalent = amount * price
        message = f'He tomado un préstamo de {amount} {ticker[:-4]} (${equivalent:.2f}).'

    logger.info(message)
    # telegram_send.send(messages=[message])
    bot.send_message(chat_id, message)

    return exit_code, price

def execute(window, ticker, prediction, best_thres, save_start, bot, client, df, dfl, dfs, first_trade):

    current_datetime = datetime.now().strftime("%d-%m at %H:%M:%S")
    cdf = 0
    coin = crypto[0][:-5]
    current_price = df['close'][-1]
    exit_code = 0
    ret = 0
    if prediction > best_thres:
        cdf = -1
    elif prediction < -best_thres:
        cdf = 1

    if dfs['Status'][0] == 0 and cdf == -1:

        # Buy/Sell
        logger.info(f'{coin} Buy Alarm Triggered')
        ret = round((current_price / dfs['Price Point'][0] - 1) * -100, 2)

        message = f'Se procede a la compra. El precio actual es de {current_price} USD.'
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit_code, dfs['Price Point'][0] = place_order(ticker, ticker[-4:], 'free', 'BUY',
                                                    'NO_SIDE_EFFECT', dfs['Price Point'][0], bot, client)

        if exit_code == 2:
            print(f'Exit Code: {exit_code}. Error de conexión')
            return dfl, dfs, first_trade

        dfs['Status'][0] = -1
        delay_tr = (datetime.now() - save_start).total_seconds()
        sl_time = window - int(delay_tr)
        if sl_time < 0:
            sl_time = 0
        time.sleep(sl_time)

        message = f'Se procede a la venta. El precio actual es de {current_price} USD.'
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit_code, dfs['Price Point'][0] = place_order(ticker, ticker[:-4], 'free', 'SELL',
                                                    'NO_SIDE_EFFECT', dfs['Price Point'][0], bot, client)

        if exit_code == 2:
            dfs['Status'][0] == -1
            print(f'Exit Code: {exit_code}. Error de conexión')
            return dfl, dfs, first_trade
        first_trade += 1

    elif dfs['Status'][0] == -1:
        message = f'Se procede a la venta. El precio actual es de {current_price} USD.'
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit_code, dfs['Price Point'][0] = place_order(ticker, ticker[:-4], 'free', 'SELL',
                                                    'NO_SIDE_EFFECT', dfs['Price Point'][0], bot, client)

        if exit_code == 2:
            print(f'Exit Code: {exit_code}. Error de conexión')
            return dfl, dfs, first_trade

    elif dfs['Status'][0] == 0 and cdf == 1 and first_trade != 0:

        # Short/Repay
        logger.info(f'{coin} Short Alarm Triggered')
        ret = round((current_price / dfs['Price Point'][0] - 1) * 100, 2)
        message = f'Se procede a shortear. El precio actual es de {current_price} USD.'
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit_code, dfs['Price Point'][0] = place_order(ticker, ticker[-4:], 'free', 'SELL', 'MARGIN_BUY',
                                                    dfs['Price Point'][0], bot, client)

        if exit_code == 2:
            print(f'Exit Code: {exit_code}. Error de conexión')
            return dfl, dfs, first_trade

        dfs['Status'][0] = 1
        delay_tr = (datetime.now() - save_start).total_seconds()
        sl_time = window - int(delay_tr)
        if sl_time < 0:
            sl_time = 0
        time.sleep(sl_time)

        message = f'Se procede a pagar deuda. El precio actual es de {current_price} USD.'
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit_code, dfs['Price Point'][0] = place_order(ticker, ticker[:-4], 'borrowed', 'BUY', 'AUTO_REPAY',
                                                    dfs['Price Point'][0], bot, client)

        if exit_code == 2:
            dfs['Status'][0] == 1
            print(f'Exit Code: {exit_code}. Error de conexión')
            return dfl, dfs, first_trade

    elif dfs['Status'][0] == 1:
        message = f'Se procede a pagar deuda. El precio actual es de {current_price} USD.'
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit_code, dfs['Price Point'][0] = place_order(ticker, ticker[:-4], 'borrowed', 'BUY', 'AUTO_REPAY',
                                                    dfs['Price Point'][0], bot, client)

        if exit_code == 2:
            print(f'Exit Code: {exit_code}. Error de conexión')
            return dfl, dfs, first_trade

    if exit_code == 3:
        message = f'Por seguridad el programa se detendrá. Por favor revisar el estado de las ordenes en el exchange y reiniciar desde el servidor.'
        logger.info(message)
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit()
    elif exit_code == 1:
        print(f'Exit Code: {exit_code}. Balance insuficiente')
        return dfl, dfs, first_trade

    dfl = append_to_col(dfl, (crypto[0]), [current_datetime, current_price, dfs['Status'][0], ret])
    dfs['Status'][0] = 0
    #log_status(dfs, cdf)

    #dfs.to_pickle(f'{loc_folder}/Status.pkl')
    dfl.to_csv(f'{loc_folder}Transactions.csv')
    dfl.to_pickle(f'{loc_folder}Transactions.pkl')
    return dfl, dfs, first_trade

def start_at_second(desired_second):
    current_second = time.localtime().tm_sec
    time_difference = desired_second - current_second

    if time_difference > 0:
        time.sleep(time_difference)

def cal_ti(df, resample_size, w0, w1, w2, w3):
    df = df.resample(resample_size).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'so': 'last',
         'pvo': 'last', 'ema': 'last', 'rsi': 'last', 'srsi': 'last', 'cci': 'last', 'psar': 'last',
         'vwap': 'last'}).fillna(
        method='ffill')
    # Calculate Stochastic Oscillator
    stochastic = StochasticOscillator(df["high"], df["low"], df["close"], window=w0)
    df["so"] = stochastic.stoch()

    # Calculate Percentage Volume Oscillator (PVO) manually
    ema_short = df["volume"].rolling(window=w1, min_periods=future_points).mean()
    ema_long = df["volume"].rolling(window=w0, min_periods=future_points).mean()
    pvo = (ema_short - ema_long) / ema_long
    df["pvo"] = pvo

    # Calculate Exponential Moving Average (EMA)
    ema = EMAIndicator(df["close"], window=w2)
    df['ema'] = (df['close'] - (ema.ema_indicator())) / df['close']

    # Calculate Stochastic RSI
    rsi = RSIIndicator(df["close"], window=w1)
    df['rsi'] = rsi.rsi()
    stoch_rsi = StochRSIIndicator(rsi.rsi(), window=w1, smooth1=w2, smooth2=w3)
    df["srsi"] = stoch_rsi.stochrsi()

    # Calculate Commodity Channel Index (CCI)
    cci = CCIIndicator(df["high"], df["low"], df["close"], window=w0)
    df["cci"] = cci.cci()

    # Calculate Parabolic SAR (PSAR)
    df['psar'] = (df['close'] - (get_psar(df, iaf=0.0002, maxaf=0.2))) / df['close']

    # Calculate the VWAP
    df['vwap'] = (df['close'] - ((df['volume'] * df['close']).cumsum() / df['volume'].cumsum())) / df['close']

    return df

def predict_return(df2, dataset_name, window, resample_size, model=model_name, scaler=scalers_name):

    price_columns = ['open', 'close', 'high', 'low']
    try:
        best_model = load_model(model)
        scalers = joblib.load(scaler)  # load input_scaler
    except:
        return df2, 0
    # ETL CSV
    dataset = pd.read_csv(dataset_name, index_col=['datetime'], parse_dates=True)
    dataset = dataset.resample(resample_size).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'so': 'last',
         'pvo': 'last', 'ema': 'last', 'rsi': 'last', 'srsi': 'last', 'cci': 'last', 'psar': 'last',
         'vwap': 'last'}).fillna(
        method='ffill')
    dataset.fillna(method='ffill', inplace=True)
    dataset.dropna(inplace=True)
    start_idx = (-1) * time_steps
    input_data = dataset.iloc[start_idx:, :].values
    feat_num = len(dataset.columns)

    # Reshape input_data to have the desired dimensions
    input_data = np.reshape(input_data, (1, time_steps, feat_num))

    # Standardize and normalize the price data
    open_data = standardize_and_normalize(input_data, scaler=scalers['open_scaler'], data_axis=0)
    high_data = standardize_and_normalize(input_data, scaler=scalers['close_scaler'], data_axis=1)
    low_data = standardize_and_normalize(input_data, scaler=scalers['high_scaler'], data_axis=2)
    close_data = standardize_and_normalize(input_data, scaler=scalers['low_scaler'], data_axis=3)

    # Standardize and normalize EMA, PSAR and VWAP
    ema_data = standardize_and_normalize(input_data, scaler=scalers['ema_scaler'], data_axis=3, p_len=4)
    psar_data = standardize_and_normalize(input_data, scaler=scalers['psar_scaler'], data_axis=7, p_len=4)
    vwap_data = standardize_and_normalize(input_data, scaler=scalers['vwap_scaler'], data_axis=8, p_len=4)

    # Normalize SO, WI, PVO, SRSI & CCI
    so_data = normalize(input_data, scaler=scalers['so_scaler'], data_axis=1, p_len=4)
    pvo_data = normalize(input_data, scaler=scalers['pvo_scaler'], data_axis=2, p_len=4)
    rsi_data = normalize(input_data, scaler=scalers['rsi_scaler'], data_axis=4, p_len=4)
    srsi_data = normalize(input_data, scaler=scalers['srsi_scaler'], data_axis=5, p_len=4)
    cci_data = normalize(input_data, scaler=scalers['cci_scaler'], data_axis=6, p_len=4)

    # Normalize the volume data
    volume_data = normalize(input_data, scaler=scalers['volume_scaler'], data_axis=4)

    # Select indicator based on the selected indexes
    arr_dic = {'so': so_data, 'pvo': pvo_data, 'ema': ema_data, 'rsi': rsi_data,
               'srsi': srsi_data, 'cci': cci_data, 'psar': psar_data, 'vwap': vwap_data}
    array = [arr_dic[arr_list[i]] for i in range(len(arr_list))]

    # Concatenate the selected arrays along the third axis
    input_sequences = np.concatenate((open_data, high_data, low_data, close_data, volume_data, *array), axis=2)

    # Make predictions
    predictions = best_model.predict(input_sequences, verbose=verbose)
    predictions = scalers['target_scaler'].inverse_transform(predictions.reshape(-1, 1))

    # Print the predictions
    prediction_time = dataset.index[-2] + timedelta(seconds=window)
    modeling_time = dataset.index[-2]
    predicted_return = predictions[0][0]
    current_price = dataset.close[-2]
    predicted_price = round(dataset.close[-1] * (1 + (predictions[0][0])), 2)

    # Create a new row with the candlestick data
    new_row = pd.DataFrame(
        {"Modeling Time": modeling_time,
         "Prediction Time": prediction_time,
         "Current Price": current_price,
         "Predicted Price": predicted_price,
         "Predicted Return %": predicted_return,
         }, index=[pd.to_datetime(datetime.now())], )

    df2 = pd.concat([df2, new_row])
    oldest_allowed_t = datetime.now() - timedelta(seconds=threshold_time)
    df2 = df2[df2.index >= oldest_allowed_t]
    df2.to_csv(f'{loc_folder}Predictions.csv', index_label='datetime')
    return df2, predictions[0][0]

