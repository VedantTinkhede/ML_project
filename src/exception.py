import sys
from src.logger import logging 

def error_message_details(error, error_detail:sys):
    _,_, exc_tb = error_detail.exc_info()  # exc_tb contains the sequence of function calls that led to the error
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name where the error occurred

    error_message = "Error occurred in script [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error) # Get the line number and error message
    ) 
    return error_message  # Return the formatted error message string


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)  # Call the base class constructor with the error message
        self.error_message = error_message_details(error_message, error_detail)  # Store the detailed error message

    def __str__(self):
        return self.error_message  # Return the detailed error message when the exception is printed


    
