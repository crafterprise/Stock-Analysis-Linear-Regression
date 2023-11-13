import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the EPS data
eps_data = pd.read_csv('eps_data_cleaned.csv')

# Add the stock price data for AAPL
stock_price_data = {
    "2008": 3.048214,
    "2009": 7.526071,
    "2010": 11.520000,
    "2011": 14.464286,
    "2012": 19.006071,
    "2013": 20.036428,
    "2014": 27.594999,
    "2015": 26.315001,
    "2016": 28.955000,
    "2017": 43.064999,
    "2018": 39.435001,
    "2019": 73.412498,
    "2020": 132.690002,
    "2021": 177.570007,
    "2022": 129.929993
}

# Title of the dashboard and introduction text
st.title('Stock Performance Analysis Dashboard')

st.markdown("""
            # Stock Analysis Project

## Overview

This project aims to analyze the stock performance and financial metrics of various companies. We specifically focus on two performance indicators - Earnings Per Share (EPS) and end-of-year stock prices. The objective is to understand the correlation, if any, between these two metrics. Another aim of the project is to demonstrate knowledge of data collection, cleaning and building a simple predictive model using linear regression.

## Data Sources

- **EPS Data:** Obtained from https://datajockey.io
- **Stock Prices:** Fetched using the `yfinance` library for historical stock prices.

## Metrics Analyzed

1. **Earnings Per Share (EPS)**: Represents the portion of a company's profit allocated to each outstanding share of common stock. It serves as an indicator of a company's profitability.

2. **End-of-Year Stock Prices**: These are the stock prices at the end of each year, serving as a snapshot of the company's market performance.

## Steps

1. **Data Collection**: Fetch the financial metrics and stock prices.

    1a. **Collecting EPS Data**: Collecting EPS data.
    
    1b. **Cleaning EPS Data**: Process the data for analysis.
    
    1c. **Collecting Stock Price Data**: Collecting Stock Price Data from yfinance.
    
    1d. **Cleaning Stock Price Data**: Process the data for analysis.
    
2. **Data Analysis**: Compute correlations and other statistical measures.
4. **Predictive Modelling**: Use Linear Regression to build a predictive model.

## Tools Used

- Python
- Pandas for data manipulation
- yfinance for fetching stock data
- matplotlib for plotting data
- scikit learn for implementing models


### Author

Kabir Inaganti

""")

# Select a company to display
company = st.selectbox('Select a company:', eps_data['Unnamed: 0'])

# Filter data for the selected company
company_data = eps_data[eps_data['Unnamed: 0'] == company].drop('Unnamed: 0', axis=1).T
company_data.columns = ['EPS']

# Create a line chart for EPS data
st.subheader(f'Historical EPS Data for {company}')
st.line_chart(company_data)

# Predictive model for Apple
if company == 'AAPL':
    st.subheader(f'Predictive Stock Price Model for {company}')



    # Append the stock price data to the EPS data for AAPL
    apple_data = company_data.copy()
    apple_data['Stock_Price'] = apple_data.index.map(stock_price_data)
    # Convert index to integer years for modeling
    years = np.array([int(year) for year in apple_data.index.values]).reshape(-1, 1)
    # Prepare the data for modeling
    X = years  # Years for training, already numeric
    y = apple_data['Stock_Price'].values  # Stock prices as target variable

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model on the entire dataset (for plotting regression line)
    model.fit(X, y)

    # Create a sequence of evenly spaced numbers over the years range for plotting
    X_plot = np.linspace(years.min(), years.max(), 100).reshape(-1, 1)
    # Predict stock prices using the sequence of years
    y_plot = model.predict(X_plot)

    # Plot the actual stock prices and the regression line
    fig, ax = plt.subplots()
    
    # Scatter plot for actual stock prices
    ax.scatter(X, y, color='blue', label='Actual Stock Price')
    
    # Line plot for predicted stock prices using the regression line
    ax.plot(X_plot, y_plot, color='red', label='Regression Line')
    
    # Add labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Stock Price')
    ax.set_title('Actual vs Predicted Stock Price for AAPL')
    
    # Add legend
    ax.legend()
    
    # Show plot
    st.pyplot(fig)

# If the user wants to see the model for another company, display a message
else:
    st.write("The predictive model is currently only available for Apple Inc. (AAPL).")

# ... [rest of the plotting code]

# Calculate the performance metrics
# Predict stock prices for the actual years
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
coef = model.coef_[0]
intercept = model.intercept_

# Add interpretation text with the model's metrics to the plot
interpretation_text = (
    f"### Interpretation:\n\n"
    f"The scatter plot shows the actual stock prices (in blue) against the earnings per share (EPS) for each year. The red line represents the predicted stock prices based on the EPS data, as modeled by our linear regression.\n\n"

f"The linear regression model attempts to draw a straight line that best fits the data points and can be used to predict the stock price given the EPS value. This line is the result of minimizing the distance between each data point and the line itself (the error).\n\n"

f"From the graph, we can observe that there is a general trend where the stock price increases as the EPS increases, which is consistent with what one might expect: as a company's earnings per share grow, its stock price also tends to rise, reflecting the company's improved profitability. However, there are some deviations from this trend, indicating that factors other than EPS also significantly influence the stock price\n\n"
      f"### Metrics:\n\n"
    f"Root Mean Square Error (RMSE): The mean squared error of the model's predictions is approximately {mse:.2f}. "
    f"This value quantifies the average squared difference between the actual and predicted stock prices, "
    f"with a lower value indicating a better fit.\n\n"
    f"Coefficient Of Determination (R^2): The coefficient of determination is approximately {r2:.2f}. "
    f"This score ranges from 0 to 1, with 1 meaning the model perfectly predicts the target variable. "
    f"An R^2 score of {r2:.2f} suggests that the model explains a significant portion of the variance in the stock price.\n\n"
    f"Model's Coefficient: The model's coefficient for EPS is approximately {coef:.2f}. "
    f"This value indicates that for each unit increase in EPS, the stock price is expected to increase by roughly {coef:.2f} units, "
    f"all else being equal."
)
# # fig, ax = plt.subplots(figsize=(10, 6))  # Specify a larger figure size
# # Show the interpretation text below the plot
# plt.figtext(0.5, 0.02, interpretation_text, ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
# # # Adjust layout to prevent clipping of ylabel
# # plt.tight_layout()

# # Show plot with interpretation
# st.pyplot(fig)

st.markdown(interpretation_text)
