import logging
import time
from datetime import datetime
import os
import sys


filepath = os.path.join(os.getcwd(), 'Logs')
filename_format = datetime.now().strftime("%m-%d-%Y %H-%M-%S")+".log" 


filenames = os.path.join(filepath, filename_format)

fmt = "[%(asctime)s] : %(name)s: %(levelname)s: %(message)s"

logging.basicConfig(filename=filenames,format=fmt ,level=logging.DEBUG)

if __name__ == "__main__":
    logging.info("sdfadsf")