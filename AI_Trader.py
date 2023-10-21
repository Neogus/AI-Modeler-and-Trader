import websocket
import json
import threading
from tensorflow.keras.models import load_model
import joblib
from binance.client import Client
import telebot
from AI_Func import *

# Set display options
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Disable column width truncation


'''
                                       ---------- AI Trader -----------

This tool will wait for a model file coming from the AI_Modeler tool in order to apply to a dataset that is constantly 
updated by the stream of info coming from the websocket connection hence creating predictions on the future return and
trading accordingly.
In order to work model and scalers files should be available in the "loc_folder". 
The program will look for the dataset and trim it based on the defined threshold time to use the recent information. 

'''

REST_API_BASE_URL = "https://api.binance.com/api/v3"
WEBSOCKET_API_BASE_URL = "wss://stream.binance.com:9443/ws"
api_key = B_API_KEY
api_secret = B_API_SECRETKEY
client = Client(api_key, api_secret)
bot = telebot.TeleBot(T_API_KEY)
ticker = crypto[0].replace("/", "")
symbol = ticker.lower()
idx = (time_steps + future_points * sequences) * validations
resample_size = f'{str(resample_co)}s'
window = future_points * resample_co  # The equivalent in seconds of the future points
w0 = 28 * future_points
w1 = int(w0/2)
w2 = int(w1/2)
w3 = int(w2/2)
limit_len = (w0 + idx) * resample_co
max_data_age = limit_len  # Maximum age of data to keep in seconds
since = round_up(limit_len/3600, 2) # Since must be expressed in hours if limit len is expressed in seconds  we divide by 3600.
first_trade = 0

# Import base candles
Import_AI(name=dataset_name, cryptos=crypto, sample_freq=interval, since=since, future_points=future_points, resample_size=resample_size)
df = pd.read_csv(dataset_name, index_col='datetime', parse_dates=True)

''' 
try:
    df = pd.read_csv(dataset_name, index_col='datetime', parse_dates=True)
except:
    print('Could not found Price Dataset, creating new dataframe...')
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
'''

try:
    df2 = pd.read_csv(f'{loc_folder}Predictions.csv', index_col='datetime', parse_dates=True)
    oldest_allowed_t = datetime.now() - timedelta(seconds=threshold_time)
    df2 = df2[df2.index >= oldest_allowed_t]
except:
    print('Could not found Predictions Dataset, creating new dataframe...')
    df2 = pd.DataFrame(
        columns=["Modeling Time", "Prediction Time", "Current Price", "Predicted Price", "Predicted Return %"])

is_stopped = False  # Flag variable to indicate if the script is being stopped intentionally
print(datetime.now())

'''
def send_m(id, mes):
    while True:
        try:
            bot.send_message(id, mes)
        except Exception as e:
            error_p = str(e)
            print("An error occurred:", error_p)
            logger.info('Telegram API can not send message. Retrying')
            time.sleep(int(error_p[-2:]))
            continue
'''



try:
    dfs = pd.read_pickle(f'{loc_folder}/Status.pkl')
except:
    dfs = pd.DataFrame(
        columns=['Status', 'Price Point'],
        index=crypto)
    dfs.fillna(0, inplace=True)
    dfs['Status'] = 0
    dfs['Price Point'] = 1
    message = f'---------------REINICIANDO-----------------'
    # telegram_send.send(
    #    messages=[message])
    #bot.send_message(chat_id, message)
    bot.send_message(chat_id, message)

try:
    dfl = pd.read_pickle(f'{loc_folder}/Transactions.pkl')
except:
    mux = pd.MultiIndex.from_product([crypto, ['Datetime', 'Price', 'Str', 'Ret']])
    dfl = pd.DataFrame(columns=mux)





def append_to_col(dft, col, val):
    idx = dft[col].last_valid_index()
    dft.loc[idx + 1 if idx is not None else 0, col] = val
    return dft

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


def ret_threshold(aux):
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

        ''' 
        # Waits for short/long signal and trades on both sides
        aux.loc[(aux['Predicted Return %'] > ret_t), 'Trade'] = -aux['Current Price']
        aux.loc[(aux['Predicted Return %'] < -ret_t), 'Trade'] = aux['Current Price']

        if not (aux['Trade'] < 0).any() or not (aux['Trade'] > 0).any():
            print(f'Thresh:{ret_t:.6f} - Not enough trades')
            continue

        aux = aux[aux['Trade'] != 0]
        aux.loc[aux.Trade.shift(1).apply(np.sign) == aux.Trade.apply(np.sign), 'Trade'] = 0
        aux = aux[aux['Trade'] != 0]
        aux = aux.loc[(aux['Trade'] < 0).idxmax():]
        perc = aux['Trade'].abs()
        ret = perc.pct_change()[1:]  # Buy long/ Sell short
        ret.loc[ret.reset_index().index % 2 != 0] = ret.iloc[:] * -1  # Buy long/ Sell short
        '''

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


def send_alarm(coin, action, strength, current_price, chat_id):
    try:

        if action == 'comprar' and strength == 'Stop Loss':
            message = f'Para evitar mayores perdidas deberías {action} {coin} y saldar el prestamo. El precio actual es de {current_price} USD.'
            # telegram_send.send(messages=[message])
            bot.send_message(chat_id, message)
        elif action == 'comprar' and strength != 'Stop Loss':
            message = f'Creo que es un momento {strength} para {action} {coin}. El precio actual es de {current_price} USD.'
            # telegram_send.send(messages=[message])
            bot.send_message(chat_id, message)
        elif action == 'vender' and strength == 'Stop Loss':
            message = f'Para evitar mayores perdidas deberías {action} {coin}. El precio actual es de {current_price} USD.'
            # telegram_send.send(messages=[message])
            bot.send_message(chat_id, message)
        elif action == 'vender' and strength != 'Stop Loss':
            message = f'Creo que es un momento {strength} para {action} {coin}. El precio actual es de {current_price} USD.'
            # telegram_send.send(messages=[message])
            bot.send_message(chat_id, message)
    except:
        logger.info('Something went wrong while trying to send messages!')


def log_status(dfs, cdf):
    logger.info(f'Status:\n{dfs} \nSignals: {cdf}')


def place_order(ticker, asset, asset_m, side, e_type, price_point):
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


def execute(prediction, best_thres, save_start):
    global dfl
    global dfs
    global first_trade
    current_datetime = datetime.now().strftime("%d-%m at %H:%M:%S")
    cdf = 0
    coin = crypto[0][:-5]
    current_price = df['close'][-1]
    exit_code = 0

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
                                                    'NO_SIDE_EFFECT', dfs['Price Point'][0])

        if exit_code == 2:
            return print(f'Exit Code: {exit_code}. Error de conexión')

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
                                                    'NO_SIDE_EFFECT', dfs['Price Point'][0])

        if exit_code == 2:
            dfs['Status'][0] == -1
            return print(f'Exit Code: {exit_code}. Error de conexión')
        first_trade += 1

    elif dfs['Status'][0] == -1:
        message = f'Se procede a la venta. El precio actual es de {current_price} USD.'
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit_code, dfs['Price Point'][0] = place_order(ticker, ticker[:-4], 'free', 'SELL',
                                                    'NO_SIDE_EFFECT', dfs['Price Point'][0])

        if exit_code == 2:
            return print(f'Exit Code: {exit_code}. Error de conexión')
    elif dfs['Status'][0] == 0 and cdf == 1 and first_trade != 0:

        # Short/Repay
        logger.info(f'{coin} Short Alarm Triggered')
        ret = round((current_price / dfs['Price Point'][0] - 1) * 100, 2)
        message = f'Se procede a shortear. El precio actual es de {current_price} USD.'
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit_code, dfs['Price Point'][0] = place_order(ticker, ticker[-4:], 'free', 'SELL', 'MARGIN_BUY',
                                                    dfs['Price Point'][0])

        if exit_code == 2:
            return print(f'Exit Code: {exit_code}. Error de conexión')

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
                                                    dfs['Price Point'][0])

        if exit_code == 2:
            dfs['Status'][0] == 1
            return print(f'Exit Code: {exit_code}. Error de conexión')

    elif dfs['Status'][0] == 1:
        message = f'Se procede a pagar deuda. El precio actual es de {current_price} USD.'
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit_code, dfs['Price Point'][0] = place_order(ticker, ticker[:-4], 'borrowed', 'BUY', 'AUTO_REPAY',
                                                    dfs['Price Point'][0])

        if exit_code == 2:
            return print(f'Exit Code: {exit_code}. Error de conexión')

    if exit_code == 3:
        message = f'Por seguridad el programa se detendrá. Por favor revisar el estado de las ordenes en el exchange y reiniciar desde el servidor.'
        logger.info(message)
        # telegram_send.send(messages=[message])
        bot.send_message(chat_id, message)
        exit()
    elif exit_code == 1:
        return print(f'Exit Code: {exit_code}. Balance insuficiente')

    dfl = append_to_col(dfl, (crypto[0]), [current_datetime, current_price, dfs['Status'][0], ret])
    dfs['Status'][0] = 0
    #log_status(dfs, cdf)

    #dfs.to_pickle(f'{loc_folder}/Status.pkl')
    dfl.to_csv(f'{loc_folder}Transactions.csv')
    dfl.to_pickle(f'{loc_folder}Transactions.pkl')
    return



def start_at_second(desired_second):
    current_second = time.localtime().tm_sec
    time_difference = desired_second - current_second

    if time_difference > 0:
        time.sleep(time_difference)


def predict_return(df2, dataset_name, model=model_name, scaler=scalers_name):

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
    epsilon = 1e-8  # small value to avoid division by zero
    feat_num = len(dataset.columns)

    # Reshape input_data to have the desired dimensions
    input_data = np.reshape(input_data, (1, time_steps, feat_num))

    # Standardize and normalize the price data
    open_data = input_data[:, :, 0]
    open_data = (open_data - np.mean(open_data, axis=0)) / (np.std(open_data, axis=0) + epsilon)
    open_data = scalers['open_scaler'].fit_transform(open_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    high_data = input_data[:, :, 1]
    high_data = (high_data - np.mean(high_data, axis=0)) / (np.std(high_data, axis=0) + epsilon)
    high_data = scalers['high_scaler'].fit_transform(high_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    low_data = input_data[:, :, 2]
    low_data = (low_data - np.mean(low_data, axis=0)) / (np.std(low_data, axis=0) + epsilon)
    low_data = scalers['low_scaler'].fit_transform(low_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    close_data = input_data[:, :, 3]
    close_data = (close_data - np.mean(close_data, axis=0)) / (np.std(close_data, axis=0) + epsilon)
    close_data = scalers['close_scaler'].fit_transform(close_data.reshape(-1, 1)).reshape(-1, time_steps, 1)

    # Standardize and normalize EMA and PSAR
    ema_data = input_data[:, :, len(price_columns) + 3]
    ema_data = (ema_data - np.mean(ema_data, axis=0)) / (np.std(ema_data, axis=0) + epsilon)
    ema_data = scalers['ema_scaler'].fit_transform(ema_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    psar_data = input_data[:, :, len(price_columns) + 7]
    psar_data = (psar_data - np.mean(psar_data, axis=0)) / (np.std(psar_data, axis=0) + epsilon)
    psar_data = scalers['psar_scaler'].fit_transform(psar_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    vwap_data = input_data[:, :, len(price_columns) + 8]
    vwap_data = (vwap_data - np.mean(vwap_data, axis=0)) / (np.std(vwap_data, axis=0) + epsilon)
    vwap_data = scalers['vwap_scaler'].fit_transform(vwap_data.reshape(-1, 1)).reshape(-1, time_steps, 1)

    # Normalize SO, WI, PVO, SRSI & CCI
    so_data = input_data[:, :, len(price_columns) + 1]
    so_data = scalers['so_scaler'].fit_transform(so_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    pvo_data = input_data[:, :, len(price_columns) + 2]
    pvo_data = scalers['pvo_scaler'].fit_transform(pvo_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    rsi_data = input_data[:, :, len(price_columns) + 4]
    rsi_data = scalers['rsi_scaler'].fit_transform(rsi_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    srsi_data = input_data[:, :, len(price_columns) + 5]
    srsi_data = scalers['srsi_scaler'].fit_transform(srsi_data.reshape(-1, 1)).reshape(-1, time_steps, 1)
    cci_data = input_data[:, :, len(price_columns) + 6]
    cci_data = scalers['cci_scaler'].fit_transform(cci_data.reshape(-1, 1)).reshape(-1, time_steps, 1)

    # Standardize the volume data
    volume_data = input_data[:, :, len(price_columns)]
    volume_data = scalers['volume_scaler'].fit_transform(volume_data.reshape(-1, 1)).reshape(-1, time_steps, 1)

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


def on_message(ws, message):
    data = json.loads(message)

    if "k" in data:
        candle = data["k"]
        timestamp = int(candle["t"]) // 1000  # Convert timestamp to seconds
        open_price = float(candle["o"])
        high_price = float(candle["h"])
        low_price = float(candle["l"])
        close_price = float(candle["c"])
        volume = float(candle["v"])

        # Create a new row with the candlestick data
        new_row = pd.DataFrame(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            },
            index=[pd.to_datetime(timestamp, unit="s")],
        )

        # Append the new row to the DataFrame
        global df
        df = pd.concat([df, new_row])

        # Remove lines older than the maximum data age
        oldest_allowed_time = datetime.now() - timedelta(hours=2) - timedelta(seconds=max_data_age)
        df = df[df.index >= oldest_allowed_time]
        # print(len(df))


def on_error(ws, error):
    print("WebSocket error:", error)


def on_close(ws, close_status_code, close_msg):
    if not is_stopped:
        print("WebSocket connection closed")
        reconnect()


def on_open(ws):
    print("WebSocket connection opened")
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params": [f"{symbol}@kline{interval}"],
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))


def reconnect():
    print("Reconnecting...")
    delay_c = 1  # Initial delay in seconds
    while True:
        try:
            time.sleep(delay_c)
            # Establish WebSocket connection
            websocket.enableTrace(False)
            ws = websocket.WebSocketApp(
                f"{WEBSOCKET_API_BASE_URL}/{symbol.lower()}@kline_{interval}",
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )
            websocket_thread = threading.Thread(target=ws.run_forever)
            websocket_thread.start()
            break  # If the connection is successful, exit the loop
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            delay *= 2  # Exponentially increase the delay
            delay += random.randint(1, 60)  # Add some randomization to the delay
            delay = min(delay, 600)  # Limit the maximum delay to 60 seconds


# Establish WebSocket connection
websocket.enableTrace(False)
ws = websocket.WebSocketApp(
    f"{WEBSOCKET_API_BASE_URL}/{symbol.lower()}@kline_{interval}",
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
    on_open=on_open,
)
websocket_thread = threading.Thread(target=ws.run_forever)
websocket_thread.start()


# Save DataFrame to CSV every minute
def save_to_csv():
    global df
    global df2

    # Wait to sync candles at the last second before saving
    save_start = datetime.now()

    # Calculate technical indicators
    print(f'Current Dataframe Length:{len(df)}')

    #if len(df) > w0 + time_steps + future_points:
    calculate_technical_indicators()
    #start_at_second(59)
    df.to_csv(dataset_name, index_label='datetime')

    #if len(df) > limit_len:
    df2, prediction = predict_return(df2, dataset_name)
    print(f'Predicted return:{prediction}')
    if len(df2) > 5:
        #best_thres = ret_threshold(df2)
        if abs(prediction) > best_thres:
            delay_t = (datetime.now() - save_start).total_seconds()
            print(f'Pre-execution delay: {delay_t}')
            execute(prediction, best_thres, save_start)  # This executes the trade!!!

    delay = (datetime.now() - save_start).total_seconds()
    print(f'Total delay: {delay}')
    threading.Timer(window - delay - 10, save_to_csv).start()


def calculate_technical_indicators():
    global df

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
    df['psar'] = (df['close'] - (get_psar(df, iaf=0.0002, maxaf=0.2)))/df['close']

    # Calculate the VWAP
    df['vwap'] = (df['close'] - ((df['volume'] * df['close']).cumsum() / df['volume'].cumsum())) / df['close']


# Start saving DataFrame to CSV
start_at_second(59)
threading.Timer(0, save_to_csv).start()

try:
    while True:
        pass
except KeyboardInterrupt:
    is_stopped = True
