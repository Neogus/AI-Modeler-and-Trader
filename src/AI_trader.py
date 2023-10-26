import websocket
import json
import threading
from tensorflow.keras.models import load_model
import joblib
from binance.client import Client
import telebot
from src.AI_fun import *
import random
import pandas as pd

# Set display options
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Disable column width truncation


'''
                                       ---------- AI Trader -----------

This tool will wait for a model file coming from the AI_Modeler tool in order to apply on a datset that is constantly 
updated by the stream of info coming from the websocket connection hence creating predictions on the future return and
trading accordingly.
In order to work model and scalers files should be available in the "loc_folder". 
The program will look for the dataset and trim it based on the defined threshold time to used the recent information. 

'''
global df
def trader():
    global df
    global df2
    global dfl
    global dfs
    global first_trade
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
    # Save DataFrame to CSV every minute

    def save_to_csv():
        global df
        global df2
        global dfl
        global dfs
        global first_trade
        # Wait to sync candles at the last second before saving
        save_start = datetime.now()
        # Calculate technical indicators
        print(f'Current Dataframe Length: {len(df)} rows')
        #if len(df) > w0 + time_steps + future_points:
        cal_ti(df, resample_size, w0, w1, w2, w3)
        #start_at_second(59)
        df.to_csv(dataset_name, index_label='datetime')
        #if len(df) > limit_len:

        df2, prediction = predict_return(df2, dataset_name, window, resample_size, model=model_name, scaler=scalers_name)
        print(f'Predicted return:{prediction}')
        if len(df2) > 5:
            #best_thres = ret_threshold(df2)
            if abs(prediction) > best_thres:
                delay_t = (datetime.now() - save_start).total_seconds()
                print(f'Pre-execution delay: {delay_t}')
                dfl, dfs, first_trade = execute(window, ticker, prediction, best_thres, save_start, bot, client, df, dfl, dfs, first_trade)  # This executes the trade!!!
        delay = (datetime.now() - save_start).total_seconds()
        print(f'Total delay: {delay}')
        threading.Timer(window - delay - 10, save_to_csv).start()

    def on_message(ws, message):
        global df
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
            df = pd.concat([df, new_row])
            # Remove lines older than the maximum data age
            oldest_allowed_time = datetime.now() - timedelta(hours=2) - timedelta(seconds=max_data_age)
            df = df[df.index >= oldest_allowed_time]
            #print(len(df))

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
    # Start saving DataFrame to CSV
    start_at_second(59)
    threading.Timer(0, save_to_csv).start()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        is_stopped = True