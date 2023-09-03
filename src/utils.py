from src.logger import logging
from src.exception import CustomException
import os
import sys
import pickle 
from sklearn.metrics import r2_score


def save_object(filepath, object):
    try:
        #logging.critical("the filepath {}".format(filepath))
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        #logging.critical("The dirpath is {}".format(dir_path))
        with open(filepath,"wb") as f:
            pickle.dump(object,f)

    except Exception as e:
        #logging.error("Error occured in save_object with object {}".format(object))
        raise CustomException(e,sys)
    

def evaluate_models(train_features, train_target, test_features, test_target, models):


    report = {}

    for i in range(len(models)):
        
        model = list(models.values())[i]

        model.fit(train_features, train_target)
        y_pred = model.predict(test_features)
        score = r2_score(y_pred, test_target)

        report[list(models.keys())[i]] = score 

        return report


def load_object(filepath):

    with open(filepath, 'rb') as f:
        return pickle.load(f)