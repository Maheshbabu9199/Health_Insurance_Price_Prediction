import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("Artifacts", "preprocessor2.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    

    def get_data_transformer_obj(self):
        try:
            logging.info("Getting data transformer")
            num_pipeline = Pipeline(steps=[("standardScaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[("OneHotEncoding", OneHotEncoder())])
            num_columns = ["age","bmi","children"]
            cat_columns = ["sex","smoker","region"]
            col_transformer = ColumnTransformer(transformers=[("num_pipeline", num_pipeline, num_columns), ("cat_pipeline", cat_pipeline, cat_columns)])
            preprocessor_pipe = make_pipeline(col_transformer)
            return preprocessor_pipe
        except Exception as e:
            logging.error("Error getting data transformer")
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Obtaining the data inside the initiate data transformation() completed")
            
            preprocessor_obj = self.get_data_transformer_obj()
            #logging.info("Got the transformer object: {}" .format(preprocessor_obj))
            target_column = "expenses"
            train_features = train_df.drop(columns=["expenses"],axis=1)
            train_target = train_df[target_column]

            test_features = test_df.drop(columns=["expenses"],axis=1)
            test_target = test_df[target_column]

            logging.info("train_df: {}, test_df: {}".format(train_df.shape, test_df.shape))

            train_arr_fromobj = preprocessor_obj.fit_transform(train_features)
            test_arr_fromobj = preprocessor_obj.transform(test_features)
            
            logging.info("train_arr_fromobj: {}, test_arr_fromobj: {}".format(train_arr_fromobj.shape, test_arr_fromobj.shape))

            train_arr = np.c_[train_arr_fromobj, np.array(train_target)]
            test_arr = np.c_[test_arr_fromobj, np.array(test_target)]

            logging.info("training arr after np.c_ is {}:".format(train_arr))
            logging.info("testing arr after np.c_ is {}:".format(test_arr))

            save_object(self.data_transformation_config.preprocessor_path, preprocessor_obj)

            return train_arr, test_arr



        except Exception as e:
            logging.error("{}".format(e))
            raise CustomException(e,sys)