"""
This Module contains multiple tests for churn_library.py
These tests are:
    * test_import
    * test_eda
    * test_encoder_helper
    * test_perform_feature_engineering
    * test_train_models
Author: Fernando Dorado Rueda
Date: March 2022
"""


import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

CATEGORY_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info(" SUCCESS: Data Imported")
    except FileNotFoundError as err:
        logging.error(
            "ERROR: Cannot be possible to import data, check data location")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Error: Empty File")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        dataframe = test_import(cls.import_data)
        perform_eda(dataframe)
        logging.info("SUCCESS: Testing EDA finished")
    except BaseException:
        logging.error("ERROR: Testing EDA failed")
        raise 

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        dataframe = test_import(cls.import_data)
        encoder_helper(dataframe, CATEGORY_COLUMNS, "Churn")
        logging.info("SUCCESS: Testing encoder_helper finished")
        return dataframe
    except BaseException:
        logging.error("ERROR: Testing encoder_helper failed")
        raise

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        dataframe = test_encoder_helper(cls.encoder_helper)
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe, "Churn")
        logging.info("SUCCESS: Testing FeasS")
        return x_train, x_test, y_train, y_test
    except BaseException:
        logging.error(
            "ERROR: Testing test_perform_feature_engineering failed")
        raise


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        x_train, x_test, y_train, y_test = test_perform_feature_engineering(
            cls.perform_feature_engineering)
        train_models(x_train, x_test, y_train, y_test)
        logging.info("SUCCESS: Testing test_train_models")
    except BaseException:
        logging.error(
            "ERROR: Testing test_train_model failed")
        raise


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)








