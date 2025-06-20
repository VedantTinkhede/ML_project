import sys
from src.logger import logging 

def error_message_details(error, error_detail:sys):
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script [{file_name}] line number [{line_number}] error message[{error}]"
    else:
        return f"Error message: {error}"  # Return the formatted error message string


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)  # Call the base class constructor with the error message
        self.error_message = error_message_details(error_message, error_detail)  # Store the detailed error message

    def __str__(self):
        return self.error_message  # Return the detailed error message when the exception is printed


    
