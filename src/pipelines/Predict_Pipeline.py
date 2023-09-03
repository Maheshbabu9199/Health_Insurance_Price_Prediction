import os
import sys
from src.logger import logging
from src.exception import CustomException 
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self, data):
        self.features_data = data
        self.expense = self.predict()

    def predict(self):
        try:
            logging.info("starting prediction")
            model_path = "Artifacts\model.pkl"
            preprocessor_path = "Artifacts\preprocessor2.pkl"
            model_obj = load_object(model_path)
            preprocessor_obj = load_object(preprocessor_path)
            print("From 21 line in predict: {}".format(type(self.features_data)))
            predict_data = preprocessor_obj.transform(self.features_data)
            model_pred = model_obj.predict(predict_data)
            
            return model_pred
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, age, gender, bmi, region, children, smoker):
        self.age = age
        self.gender = gender
        self.bmi = bmi
        self.region = region
        self.children = children
        self.smoker = smoker

    
    def get_dataframe(self):
        try:
            

            temp_dict = {"age": [self.age], "sex": [self.gender],  "bmi": [self.bmi], "children": [self.children], "smoker": [self.smoker], "region": [self.region]}

            predict_df = pd.DataFrame(temp_dict)
            logging.error("predict_df shape: {}".format(predict_df.shape))
            return predict_df
        except Exception as e:
            logging.error("Error getting dataframe in predictpipeline")
            raise CustomException(e,sys)

        