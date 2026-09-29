import sys
import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info() 
    ## exc_tb will give info about where the error has occurred in the code. It will give us the file name and line number where the error has occurred.
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
        return error_message
        
        )
class customException (Exception) 
    def__init__(self, error_message , error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def__str__(self):
        return self.error_message


