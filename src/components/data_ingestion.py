import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_development import ModelDevelopment, ModelDevelopmentConfig

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("Artifacts","train_data.csv")
    test_data_path = os.path.join("Artifacts","test_data.csv")
    raw_data_path  = os.path.join("Artifacts","raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv('Notebooks\insurance (1).csv')
            logging.debug("Dataset read from the notebooks folder")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=0)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            
            return self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path


        except Exception as e:
            logging.error("Got error in initiate data ingestion place")
            raise CustomException(e,sys)
        

if __name__== "__main__":
    data_obj = DataIngestion()
    train_path, test_path = data_obj.initiate_data_ingestion()
    data_transformation_obj = DataTransformation()
    train_dataset, test_dataset = data_transformation_obj.initiate_data_transformation(train_path, test_path)
    print(train_dataset.shape, test_dataset.shape)
    model_obj = ModelDevelopment()
    model_obj.initiate_model_training(train_dataset, test_dataset)