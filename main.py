import os
import logging
from src.AI_Modeler import main_modeler
import src.AI_Config

if __name__=="__main__":
    
    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logger.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M', filename=file, filemode='a')
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger('Log')
    main_modeler()