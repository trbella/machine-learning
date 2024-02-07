# Hotel Cancellations Analysis

This repository contains files related to the analysis of hotel cancellations. The analysis aims to uncover insights and build a predictive model to assist in preventing cancellations and taking timely retention actions.

## Files

### 1. catboost_no_timeseries.pkl

This file contains a trained CatBoost classifier model without considering time series data. The model is optimized to predict hotel cancellations based on various features without considering temporal patterns.

### 2. catboost_timeseries.pkl

This file contains a trained CatBoost classifier model specifically considering time series data. Unlike the previous model, this one takes into account temporal patterns and trends to improve the prediction of hotel cancellations.

### 3. hotel_cancellations.ipynb

This Jupyter Notebook file contains the code used for the analysis, hypothesis development, data visualization, model building, and evaluation. It includes detailed explanations and insights derived from the data exploration and modeling process.

### 4. profiling_report.html

This HTML file contains a profiling report generated using a profiling tool. The report provides detailed statistical information and insights into the dataset, helping to understand its structure, distributions, and potential issues.

## Usage

1. **Model Deployment**: The trained CatBoost models (catboost_no_timeseries.pkl and catboost_timeseries.pkl) can be deployed for predicting hotel cancellations in real-time or batch processing scenarios.

2. **Analysis and Visualization**: Explore the Jupyter Notebook (hotel_cancellations.ipynb) to understand the data, visualize key trends, develop hypotheses, and evaluate the predictive models.

3. **Profiling Report**: Refer to the profiling report (profiling_report.html) for comprehensive statistical analysis and insights into the dataset.

## Note

For detailed instructions on how to use the models and interpret the analysis, please refer to the Jupyter Notebook (hotel_cancellations.ipynb).
