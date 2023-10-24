# AI-Modeler-and-Trader
Tools for modeling and deploying recurrent long-short-term memory neural networks for price movement forecast and trading

1 - Introduction

This repository contains a series of programs to train/test an LSTM neural network to predict price direction and use it to trade algorithmically. The folder "src"  contains the configuration files and the modules that will be called by the main. The main will ask us to select the model we would like to run (Tuner, Modeler, or Trader) while "AI_Config.py" contains the shared configuration of these tools. Lastly, AI_Func is a function library used by the scripts and c_config.py contains the API keys and Chat ID variable necessary to run the trader, send messages to a telegram chat, and place orders in the Binance exchange. 
Before executing the main make sure to set up the "AI_Config" file and keep it open for further changes.
The Neural Network will input a number of features (OHLCV and a combination of technical indicators) to target the return and try to forecast price movement.

2 - Tuner

After adjusting the LSTM variables in AI_Config, this module will download a dataset from Binance exchange and tune the hyperparameters of the network based on the base OHLCV dataset and a fixed or random combination of indicators as features, defined in the list "arr_list" inside AI_Config.py.
The program will output the results on the screen, showing the average accuracy obtained, hyperparameters, and the tested combination. The hyperparameters with the best result should be manually re-introduced in the default hyperparameter section in the "AI_Config" file along with the combination of indicators defined in the variable "arr_list". 

3 - Modeler

The modeler will use the parameters defined in AI_Config to train/test an RNN LSTM based on the modeler_mode. If modeler_mode = "solo" the modeler will download its own dataset and use it for modeling otherwise if modeler_mode = "parallel" mode the program will wait for the trader to download its dataset and use this dataset to model. After finding a better model evaluated by its accuracy in predicting the price movement, the program will save a model file and scaler file that will be used by the trader.

4 - Trader

The trader will establish a WebSocket connection with the Binance exchange and update a live data frame, at the same time it will use the files provided by the modeler to predict return and then if conditions are met execute the trade by placing orders in the exchange.

5 - Legal 

5.1 Proper Use: Gustavo Rabino (hereinafter referred to as "the Developer") provides these programs for your use. The programs are subject to the terms and conditions of the license agreement contained in the accompanying license file. By using these programs, you agree to abide by the terms of this license agreement.

5.2 Improper Use: The Developer does not condone or support the improper use of the Program. Improper use includes, but is not limited to, any use that violates applicable laws or regulations, infringes upon intellectual property rights, or causes harm to individuals or entities.

5.3 No Responsibility: The Developer disclaims all responsibility for any improper use of the Program by any party. You acknowledge that the Developer is not liable for any actions, consequences, or damages resulting from the improper use of these programs.

5.4 License Agreement: To understand the rights and restrictions associated with the use of the Program, you must carefully read and comply with the license agreement contained in the license file accompanying the Program. If you do not agree with the terms of the license agreement, you should not use these programs.

5.5 Contact Information: If you have any questions or concerns regarding these programs, the license agreement, or this disclaimer, please contact the Developer at gusrab@gmail.com

By using the Program, you acknowledge that you have read and understood this disclaimer and the associated license agreement. You agree to use the Program in accordance with the terms and conditions specified in the license agreement.

Gustavo Rabino
24th of October, 2023
