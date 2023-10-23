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
import warnings
from src.AI_Config import * #NOT RECOMMENDED: you loose track of what you are importing, and makes it hard to track and debug.
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
        if bull:
            df.psar.iloc[i] = df.psar.iloc[i - 1] + af * (hp - df.psar.iloc[i - 1])
        else:
            df.psar.iloc[i] = df.psar.iloc[i - 1] + af * (lp - df.psar.iloc[i - 1])


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

def calculate_technical_indicators(df, w, w0, w1, w2, w3,resample_size):

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
    dfx = calculate_technical_indicators(dfx, future_points, w0, w1, w2, w3, resample_size)
    dfx.to_csv(name)
    print(f'{cryptos[0]} Historical Length: {len(dfx)}')
