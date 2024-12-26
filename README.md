# Aviachipta-narxini-bashorat-qilish
Parvoz chiptalari narxini taxmin qilish


This code performs a regression task using XGBoost, a popular machine learning model, to predict flight prices based on various features. Below is a general explanation of the steps:

### 1. **Data Loading:**
   - **`train_data` and `test_data`:** These variables load training and test datasets using pandas' `read_csv` function.
   - The datasets are expected to contain features such as flight duration, airline, source city, destination city, stops, price, etc., which will be used to train and evaluate the model.

### 2. **Feature Engineering:**
   - **Log Transformation of 'duration':** The 'duration' feature is log-transformed to deal with skewness in the distribution.
   - **Encoding 'stops':** The categorical 'stops' feature (with values 'zero' and 'one') is numerically encoded.
   - **Frequency Encoding for Categorical Variables:** The 'airline', 'source_city', and 'destination_city' categorical features are encoded based on their frequency in the training data (i.e., the proportion of each category).
   - **One-Hot Encoding:** The 'departure_time', 'arrival_time', and 'class' categorical features are one-hot encoded, converting them into binary features.
   - **Dropping Unnecessary Columns:** Several columns, including 'flight', 'duration', 'stops', 'airline', 'source_city', 'destination_city', and 'price' (target variable), are dropped from the features. Only relevant features remain.
   - **Align Test Data:** To ensure consistency between the training and test data, the test dataset is aligned with the training dataset by ensuring the same columns are present in both, filling any missing columns with zeros.

### 3. **Train-Test Split:**
   - **`train_test_split`:** The training data (`X_train` and `y_train`) is further split into training and validation sets (80% training, 20% validation) to evaluate the model's performance.

### 4. **Prepare Data for XGBoost:**
   - **DMatrix Creation:** XGBoost works with a specific data structure, `DMatrix`. This data structure is created for the training split (`X_train_split` and `y_train_split`) to efficiently train the model.

### 5. **Hyperparameter Tuning with Optuna:**
   - **Optuna Study:** The code uses Optuna, an optimization framework, to perform hyperparameter tuning for XGBoost. The `objective` function defines the hyperparameters to tune (e.g., learning rate, max depth, regularization terms), and Optuna tries different combinations to find the best one.
   - **Cross-Validation:** During each trial, XGBoost's cross-validation is used to evaluate the model with the given hyperparameters. The best hyperparameters are those that minimize the root mean squared error (RMSE).
   - **Best Parameters:** After optimizing for 50 trials, the best hyperparameters found by Optuna are printed.

### 6. **Model Training:**
   - **XGBRegressor:** With the optimized hyperparameters, an `XGBRegressor` model is trained using the full training data (`X_train_split` and `y_train_split`).
   - The model is fitted with the best hyperparameters found from Optuna.

### 7. **Model Evaluation:**
   - **Prediction on Validation Set:** The model is used to predict the prices on the validation set (`X_val_split`).
   - **Metrics Calculation:** 
     - Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are calculated to measure the model's accuracy.
     - R² score is also calculated to evaluate the model's fit to the data.
   - The RMSE and R² score are printed to assess model performance.

### 8. **Residual Analysis:**
   - **Residual Plot:** A scatter plot is created to visualize the residuals (differences between actual and predicted values) against the actual prices. This helps to identify patterns in the model's errors, with a red line indicating the zero residual line.

### 9. **Test Predictions:**
   - **Prediction on Test Set:** The trained model is used to make predictions on the test dataset (`X_test`).
   - **Save Predictions:** The test predictions are saved into a CSV file (`test_predictions.csv`), which includes the 'id' column from the test dataset and the predicted price.

### 10. **Summary:**
   - The code loads datasets, preprocesses features, tunes the XGBoost model's hyperparameters using Optuna, trains the final model, evaluates it, performs residual analysis, and then makes predictions on a test set. The test predictions are saved to a CSV file for further use.

In essence, the code automates the process of preparing data, tuning a machine learning model, evaluating its performance, and making predictions.
