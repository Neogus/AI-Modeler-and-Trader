# AI-Modeler-and-Trader
Tools for modeling and deploying recurrent long-short-term memory neural networks for price movement forecast and trading

1 - Introduction

This repository contains a series of programs to train/test an LSTM neural network to predict price direction and use it to trade algorithmically. Inside the repository, there should be 6 files. "AI_Tuner.py", "AI_Modeler.py" and "AI_Trader.py" can be used to execute the correspondent tools while "AI_Config.py" contains the configuration of these tools. Lastly, AI_Func is a function library used by the scripts and c_config.py contains the API keys and Chat ID variable necessary to run the trader, send messages to a telegram chat, and place orders in the Binance exchange. 
The Neural Network will input a number of features (OHLCV and a combination of technical indicators) to target the return.

2 - Tuner

After adjusting the LSTM variables in AI_Config, this program will download a dataset from Binance exchange and tune the hyperparameters of the network based on the base OHLCV dataset and a fixed or random combination of indicators as features, created from the list called "arr_list" inside AI_Config.py.
It will output a file (best_params.joblib) containing these parameters to be used by the modeler. The modeler will use this file and the combination of indicators defined in the variable "arr_list" inside "AI_Config". For this reason if we are not using all the indicators inside this list due to having found a different combination that uses fewer indicators then is it is necessary to change the list before executing the modeler.

3 - Modeler

The modeler will use the best_params file from the tuner and the desire combination of indicators in arr_lost in AI_config to train/test a RNN LSTM based on the modeler_mode. If modeler_mode = "solo" the modeler will download its own dataset and use it for modeling otherwise if modeler_mode = "parallel" mode the program will wait for the trader to download its dataset and use this dataset to model. After finding a better model evaluated by its accuracy in predicting the price movement, the program will save a model file and scaler file that will be used by the trader.

4 - Trader

The trader will establish a WebSocket connection with the Binance exchange and update a live data frame, at the same time it will use the files provided by the modeler to predict return and then if conditions are met execute the trade by placing orders in the exchange.
