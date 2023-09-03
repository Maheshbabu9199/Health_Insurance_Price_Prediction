import sys
import os
from src.logger import logging


""" The reason why we need this function is in normal exception we only get the error message,
    but in this error_detail it is from sys library. So, any error occurs it is recorded in sys
    To access that error we are using this function. """

def error_message_details(error,error_detail:sys):
    _, _ , exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in file [{0}] at line_no: [{1}] with message [{2}]".format(filename,exc_tb.tb_lineno,error) 
    logging.error("{}".format(error_message))
    return error_message

# This is custom class for handling exceptions

class CustomException(Exception):
    def __init__(self,error_message,error_detail):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
