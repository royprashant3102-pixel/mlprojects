import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
## it is used to craete a class variable

@dataclass #is a decorator that is used to automatically generate special methods for the class, such as __init__(), __repr__(), and __eq__(). It is used to create classes that are primarily used to store data, without having to write boilerplate code for these methods.
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv') #it will create a path for train data
    test_data_path: str = os.path.join('artifacts', 'test.csv') #it will create a path for test data
    raw_data_path: str = os.path.join('artifacts', 'raw.csv') #it will create a path for raw data
    #it helps to create a path for train, test and raw data in the artifacts folder and its output will be saved in this folder

class Dataingestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def intiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv') #it will read the csv file and store it in a dataframe
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            #it will create a directory for train data path if it does not exist

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        except Exception as e:
            raise Customexception(e, sys)

if __name__ == "__main__":
    obj = Dataingestion()
    obj.intiate_data_ingestion()