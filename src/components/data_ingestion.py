import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# dataclass is used to create a class variable
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer


# @dataclass is a decorator that automatically generates special methods for the class,
# such as __init__(), __repr__(), and __eq__(). It is used to create classes that are
# primarily used to store data, without having to write boilerplate code for these methods.
@dataclass
class DataIngestionConfig:
    # Creates paths for train, test and raw data inside the artifacts folder;
    # the output of data ingestion will be saved here

    # Path for train data
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    # Path for test data
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    # Path for raw data
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        # Load config so we know where to save train, test and raw data
        self.ingestion_config = DataIngestionConfig()

    # Fixed typo: was "intiate_data_ingestion"
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the csv file and store it in a dataframe
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe")

            # Create the artifacts directory if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data as a copy in the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split data: 80% train, 20% test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets as CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return file paths so the next step (data transformation) can use them
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Step 1: Data ingestion - read data and split into train/test CSVs
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Step 2: Data transformation - returns train array, test array, and preprocessor path
    data_transformation = DataTransformation()
    # "_" ignores the third value (preprocessor file path) since we don't need it here
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Step 3: Model training - trains all models and prints the best R2 score
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))