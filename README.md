
# Financial Market Data Analysis Toolkit: Part 5 - ML Directional Forecasting

**Author:** Siddhant Dhoot
**Date:** June 28, 2025

## Description

This project frames financial market prediction as a binary classification problem: will the market go "up" or "down" the next day? It uses historical market data for an index like the S&P 500 (`^GSPC`) to engineer a comprehensive set of features. Three different supervised machine learning models are then trained and evaluated to predict the next day's market direction.

The notebook follows a complete, end-to-end machine learning workflow, including:
* Hypothesis generation and problem framing.
* Creative feature engineering using returns, lagged returns, volatility (standard deviation), and feature crosses.
* Data preprocessing, cleaning, and scaling.
* Training and evaluating three distinct models: Logistic Regression (as a baseline), a Keras Deep Neural Network (DNN), and an XGBoost Classifier.
* Evaluating model performance using metrics like Accuracy, Precision, Recall, and F1 Score on unseen data.

## Performance Summary & Key Insights

The final results are highly instructive and align with the challenges commonly faced in quantitative financial modeling. The performance on the hold-out test set is summarized below:

| Metric                | Logistic Regression (Baseline) | Keras DNN (Tuned) | XGBoost (Gradient Boosting) |
| :-------------------- | :----------------------------- | :---------------- | :-------------------------- |
| **Accuracy** | **~54.5%** | ~51.1%            | ~51.4%                      |
| **Precision** | ~54.5%                         | **~55.2%** | **~55.2%** |
| **Recall** | **~99.5%** | ~57.3%            | ~55.9%                      |
| **F1 Score** | **~70.4%** | ~56.0%            | ~55.5%                      |

### Interpretation of Results

1.  **The Baseline Prevails:** The simplest model, Logistic Regression, achieved the highest overall Accuracy and F1 Score. Its strategy was heavily biased, essentially learning the market's slight historical upward drift and predicting "up" most of the time. This resulted in an extremely high Recall but mediocre Precision.
2.  **Complexity vs. Performance:** The more complex, non-linear models (Keras DNN and XGBoost) did not outperform the baseline. They achieved slightly better Precision, but their more balanced approach resulted in lower overall F1 scores. This demonstrates that more model complexity does not guarantee better performance on noisy data.
3.  **The "Efficient Market" Hypothesis in Practice:** This outcome is a powerful demonstration of the difficulty of predicting an efficient and liquid market index. The results strongly suggest that using only historical price and volatility data provides a very faint signal for next-day direction.

## Technologies & Libraries Used
* Python 3.x
* Jupyter Notebook
* yfinance
* pandas & numpy
* scikit-learn (for `train_test_split`, `LogisticRegression`, `StandardScaler`, `class_weight`)
* TensorFlow & Keras (for the DNN model)
* XGBoost
* matplotlib & seaborn

## How to Run This Project
1.  **Prerequisites:** Ensure you have Python 3 installed, preferably within a virtual environment.
2.  **Install Libraries:** Open your terminal and install the required libraries:
    ```bash
    pip install yfinance pandas numpy matplotlib tensorflow keras seaborn scikit-learn xgboost jupyterlab
    ```
3.  **Launch Jupyter:** In your terminal, navigate to the project directory and launch Jupyter.
4.  **Open and Run:** Open the `.ipynb` file and run the cells sequentially. You will be prompted to enter a stock ticker and a date range for the analysis.

