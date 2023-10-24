import os
import logging
from src.AI_Tuner import tuner
from src.AI_Modeler import modeler
from src.AI_Trader import trader

if __name__=="__main__":

    while True:

        selection = input('Please type in the module you would like to run:\n')
        if selection == 'tuner':
            tuner()
            exit()
        elif selection == 'modeler':
            modeler()
            exit()
        elif selection == 'trader':
            trader()
            exit()
        else:
            print(f'"{selection}" is not a valid input please type one of the valid modules (tuner, modeler or trader)')
