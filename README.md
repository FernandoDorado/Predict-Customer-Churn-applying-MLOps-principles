# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Customer retention is one of the primary KPI for companies with a subscription-based business model. Competition is tough particularly in the SaaS market where customers are free to choose from plenty of providers. One bad experience and customer may just move to the competitor resulting in customer churn. Customer churn is the percentage of customers that stopped using your company’s product or service during a certain time frame. One of the ways to calculate a churn rate is to divide the number of customers lost during a given time interval by the number of active customers at the beginning of the period. For example, if you got 1000 customers and lost 50 last month, then your monthly churn rate is 5 percent.
Predicting customer churn is a challenging but extremely important business problem especially in industries where the cost of customer acquisition (CAC) is high such as technology, telecom, finance, etc. The ability to predict that a particular customer is at a high risk of churning, while there is still time to do something about it, represents a huge additional potential revenue source for companies.

The primary objective of the customer churn predictive model is to retain customers at the highest risk of churn by proactively engaging with them. For example: Offer a gift voucher or any promotional pricing and lock them in for an additional year or two to extend their lifetime value to the company.

## Files in the Repo
- data
    * bank_data.csv : Data used in this project (21 columns).

- images
    * eda
        *   churn.png : Histogram of the target ("churn") column.
        *   customer_Age.png : Historgram of the age of the people in the dataset.
        *   heatmap_df_corr.png : Correlation matrix (in heatmap format) between the columns in the dataset. 
        *   marital_status.png : Histogram of the marital status in the dataset.
        *   total_transaction_distribution.png : Histogram of the transactions in the dataset. 
    *   results
        *   rfc_feature_importance.png feature_importance_graph.png : Feature importance by column respect to the target.
        *   training_results.png : Training results in both Random Forest and Logistic Regression (ROC and AUC included).
-   logs
    *   churn_library.log : Generated logs during the execution. 
-   models
    *   logistic_model.pkl : Best Logistic Regression model. 
    *   rfc_model.pkl : Best Random Forest model. 
-   churn_library.py : Main code of the project. It contains all the required functions to complete EDA, FeaS and Train models. 
-   churn_notebook.ipynb: Notebook with different analysis, plots, and training process. 
-   churn_script_logging_and_tests.py : Script with different tests to ensure the correct performance of the functions in churn_library.py
-   README.md : README with all the instructions and details about the project. 


## Running Files
By following the instructions below it is possible to run the project. The first step is to install the necessary dependencies

```pip install -r requirements.txt```

The script will use churn_library.py and perform the analysis. EDA plots will be stored in the path 'images/eda'. The results plots will appear in 'images/results'. In this path, you can find the best saved models.

```python churn_library.py```

In case of modifications, you can carry out some tests using the following code in CMD:

```python churn_script_logging_and_tests.py```


