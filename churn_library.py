# library doc string
"""
This module provides a number of functions used to carry out common data science tasks
in data science
This file can also be imported as a module, and contains the following functions
functions:
    - import_data - returns dataframe for the csv found in pth.
    - perform_eda - performs EDA on df and saves the figures in the images folder
    - encoder_helper - helper function for converting each categorical column into a new column with churn provision for each category
    - perform_feature_engineering - performs feature engineering on the imported dataset
    - classification_report_image - produces a classification report for the results of test results and stores the report as an image in the images folder
    - feature_importance_plot - creates and stores the feature importances
    - train_models - trains, stores model results
Author: Fernando Dorado Rueda
Date: March 2022
"""

# import libraries
import logging
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import pandas as pd
import numpy as np
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


# Storage Locations for Models
RANDOM_FOREST_MODEL = './models/rfc_model.pkl'
LOGISTIC_REGRESSION_MODEL = './models/logistic_model.pkl'

# Image Location
EDA_IMAGE_PATH = "images/eda/"
RESULT_IMAGE_PATH = "images/results/"

# Category columns from the dataframe
CATEGORY_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

# Quantity columns from the dataframe
QUANTITY_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        dataframe = pd.DataFrame(pd.read_csv(pth))
        logging.info(
            f"SUCCESS: Successfully read CSV file with a shape of %s",
            dataframe.shape)
    except FileNotFoundError as err:
        logging.error("ERROR: Failed to read CSV file at this path %s", pth)
        raise err
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("ERROR: Empty dataframe")
        raise err

    # create churn column using lambda on Attrition_flag column
    try:
        assert 'Attrition_Flag' in dataframe.columns
        assert dataframe["Attrition_Flag"].shape[0] > 0

        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        return dataframe

    except AssertionError as err:
        logging.error("ERROR: Creation of Churn Column Failed")
        raise err


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
    output:
            None
    '''
    try:
        # Specify histogram plots
        histogram_plot_columns = [
            "Churn",
            "Customer_Age",
            "Marital_Status",
            "Total_Trans_Ct",
            "Heat_Map"]

        # Create plot for the different columns
        for column in histogram_plot_columns:

            # Ignore Assertion for Heat Map
            if column != "Heat_Map":
                assert column in df.columns
                assert df[column].shape[0] > 0

            logging.info("Creating plot for %s", column)

            plt.figure(figsize=(20, 10))
            if column in CATEGORY_COLUMNS:
                df[column].value_counts('normalize').plot(kind='bar')
            elif column == "Total_Trans_Ct":
                sns.distplot(df['Total_Trans_Ct'])
            elif column == "Heat_Map":
                sns.heatmap(
                    df.corr(),
                    annot=False,
                    cmap='Dark2_r',
                    linewidths=2)
            else:
                df[column].hist()

            # Save plot
            plot_saved_location = f'{EDA_IMAGE_PATH}{column}_plot.png'
            logging.info(
                "Saving plot for %s in %s",
                column,
                plot_saved_location)
            plt.savefig(plot_saved_location)
            logging.info("SUCCESS: EDA images successfully Generated")
    except AssertionError as err:
        logging.error("ERROR: Creation of EDA images failed")
        raise err


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    try:
        for category in category_lst:
            assert category in df.columns
            assert df[category].shape[0] > 0

            logging.info(
                "Generating churn proportion for %s column",
                category)
            category_group = df.groupby(category).mean()[response]
            df[f'{category}_{response}'] = df[category].apply(
                lambda x: category_group.loc[x])

        logging.info("SUCCESS: Encoding of categorical data finished")
    except AssertionError as err:
        logging.error(
            "ERROR: Encoding of categorical data provided some errors")
        raise err


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # store the quantity columns
    keep_columns = QUANTITY_COLUMNS[:]

    # Extend with churn categorical columns
    keep_columns.extend(
        [f'{column}_{response}' for column in CATEGORY_COLUMNS])

    try:
        assert response in df.columns
        assert df[response].shape[0] > 0

        y_df = df[response]

        for column_name in keep_columns:
            assert column_name in df.columns
            assert df[column_name].shape[0] > 0

        xk_df = pd.DataFrame()
        xk_df[keep_columns] = df[keep_columns]

        # train test split
        x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(
            xk_df, y_df, test_size=0.3, random_state=42)

        logging.info("SUCCESS: Feature Engineering (FeaS) performed")
        return x_train_df, x_test_df, y_train_df, y_test_df
    except AssertionError as err:
        logging.error("ERROR: Feature engineering (FeaS) report some errors")
        raise err


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    output:
             status: strin, return if successfull
    '''
    # Create and save classification report (Random Forest)
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/classification_report_rf.png')

    # Create and save classification report (Logistic Regression)
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/classification_report_lr.png')

    return None


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate Feature Importance
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 20))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars in y-label
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names in x-label
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plot_location = f'{output_pth} feature_importance_graph.png'

    # Save plot
    plt.savefig(plot_location)

    logging.info(
        "SUCCESS: Saving graph for feature importance in %s",
        plot_location)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Create Random Forest Classifier
    rfc = RandomForestClassifier(random_state=42)
    # Create Logistic Regression Classifier
    lrc = LogisticRegression()
    # Set parameters for grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    # Perform Grid Search
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    # Fit the Random Forest Classifier
    cv_rfc.fit(X_train, y_train)
    # Fit the Logistic Linear Regression Classifier
    lrc.fit(X_train, y_train)
    # Perform test predictions with Random Forest Classifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    # Perform Test predictions with Linear Regression Classifier
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Calculate and print result scores for both classifiers
    print('Random Forest Results')
    print(f'Test Results: {classification_report(y_test, y_test_preds_rf)}')
    print(f'Train Results: {classification_report(y_train, y_train_preds_rf)}')

    print('Logistic Regression Results')
    print(f'Test Results: {classification_report(y_test, y_test_preds_lr)}')
    print(f'Train Results: {classification_report(y_train, y_train_preds_lr)}')

    # Save classification report image
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Plot the Linear Logistic Regression ROC curve and save it
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    # Plots
    plt.figure(figsize=(15, 8))
    a_x = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=a_x,
        alpha=0.8)
    lrc_plot.plot(ax=a_x, alpha=0.8)
    plt.savefig("./images/results/training_results.png")

    # Save the best models
    feature_importance_plot(
        cv_rfc,
        X_test,
        './images/results/rfc_feature_importance.png')
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Load the saved models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # Plot ROC curve
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(16, 8))
    a_x = plt.gca()
    plt.savefig("ROC_curve_lrc.png")

    # Plot and Save the Random Forest Classifier ROC
    plot_roc_curve(rfc_model, X_test, y_test, ax=a_x, alpha=0.8)
    lrc_plot.plot(ax=a_x, alpha=0.8)
    plt.savefig("ROC.png")

    return 'Completed!'


if __name__ == "__main__":
    try:
        # Create dataframe from file
        df = import_data("data/bank_data.csv")
        logging.info("SUCCESS: Data Imported")

        # Carry out EDA on the Dataframe
        perform_eda(df)
        logging.info("SUCCESS: EDA Finsished")

        # Encode Categorical Data
        encoder_helper(df, CATEGORY_COLUMNS, "Churn")
        logging.info("SUCCESS: Encoder Finsished")

        # Perform Feature engineering to split data
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, "Churn")

        logging.info("SUCCESS: FeaS Finsished")

        # Train models
        train_models(x_train, x_test, y_train, y_test)
        logging.info("SUCCESS: Trained Completed")

    except BaseException:
        logging.error("ERROR: Model Training Failed")
        raise
