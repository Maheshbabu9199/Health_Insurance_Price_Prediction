import os
import sys
from src.utils import save_object, evaluate_models
from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from dataclasses import dataclass
from sklearn.metrics import r2_score

@dataclass
class ModelDevelopmentConfig:
    model_path = os.path.join("Artifacts","model.pkl")

class ModelDevelopment:
    def __init__(self):
        self.model_path_config = ModelDevelopmentConfig()
    
    def initiate_model_training(self,train_arr, test_arr):

        train_arr_features = train_arr[:,:-1]
        train_arr_target = train_arr[:,-1]
        test_arr_features = test_arr[:,:-1]
        test_arr_target = test_arr[:,-1]

        models = {
            "linearregression": LinearRegression(),
            "decisiontree": DecisionTreeRegressor(),
            "randomforest": RandomForestRegressor(),
            "kneighbors" : KNeighborsRegressor(),
            "xgboost" : XGBRegressor()
        }

        model_report = evaluate_models(train_features = train_arr_features, train_target = train_arr_target, 
                    test_features = test_arr_features, test_target = test_arr_target, models = models)
        


        best_score = max(sorted(list(model_report.values())))

        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]

        best_model = models[best_model_name]

        if best_score < 0.65 :
            raise CustomException("No best model found")
        logging.info("Best model found")


        save_object(self.model_path_config.model_path, best_model)

        print("Best r2_score obtained: {}".format(r2_score(best_model.predict(test_arr_features), test_arr_target)))

        #return "Completed from model development"