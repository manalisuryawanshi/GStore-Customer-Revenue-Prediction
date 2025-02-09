---

# GStore Customer Revenue Prediction

This project aims to analyze customer data from the Google Merchandise Store (GStore) to predict transaction revenue per customer. The analysis leverages the Google Analytics data and applies various data preprocessing techniques, including flattening JSON-like columns, dropping irrelevant features, and filtering for relevant data points. The goal is to better understand customer behavior and assist businesses in making data-driven decisions regarding marketing budgets and operational strategies.

## Project Overview
- **Dataset**: The dataset used for this analysis comes from the Google Merchandise Store, containing customer-level data from their Google Analytics.
- **Problem Statement**: Predict the revenue per customer to help businesses optimize marketing strategies, focusing on the 80/20 rule where a small percentage of customers generate most of the revenue.
  
## Key Steps:
1. **Data Preprocessing**:
   - Dropping unnecessary columns (e.g., `sessionId`, `socialEngagementType`).
   - Flattening JSON-like columns (`totals`, `geoNetwork`, `device`, `trafficSource`).
   - Filtering out rows with missing revenue data and irrelevant columns.

2. **Feature Engineering**:
   - Creating a reduced set of features including `channelGrouping`, `device`, `trafficSource`, etc.
   - Performing one-hot encoding for categorical variables.

3. **Data Splitting**:
   - Splitting the data into training and testing sets (80%/20%).
   - Grouping transaction revenue by customer to predict total revenue per customer.

4. **Modeling**:
   - Training a Linear Regression model to predict revenue.
   - Evaluating the model performance using RMSE and R² score.

5. **Exploration and Visualization**:
   - Log-transforming the revenue data for better model performance.
   - Analyzing the distribution of non-zero revenues.

## Requirements
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `lightgbm`
  
## Files:
- `flattened_google_analytics.csv`: Data with flattened JSON columns.
- `filtered_google_analytics.csv`: Data filtered to contain non-null transaction revenue.
- `google_analytics_train.csv` and `google_analytics_test.csv`: Train and test data splits.
- `filtered_google_analytics_final.csv`: Final dataset with unnecessary columns dropped.

## Results
- Model evaluation using RMSE and R² score for predicting transaction revenue.
- Insights into customer revenue distribution and behavior.

---
