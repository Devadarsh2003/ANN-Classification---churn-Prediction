### ANN-Classification---churn-Prediction

## Overview
This project implements a customer churn prediction model using machine learning techniques. The application is built with Streamlit, allowing users to input customer data and receive predictions on whether a customer is likely to churn.

## Files Included
app.py: The main Streamlit application file that handles user input and displays predictions.
prediction.ipynb: A Jupyter Notebook containing the prediction logic and model evaluation.
experiments.ipynb: A Jupyter Notebook for experimenting with different models and parameters.

## Requirements
To run this application, you need the following Python packages:
> streamlit
> numpy
> pandas
> tensorflow
> scikit-learn
> pickle

You can install the required packages using pip:
bash
" pip install streamlit numpy pandas tensorflow scikit-learn" 
or 
" pip install requirements.txt "

## How to Run the Application
1. Ensure you have the trained model (model.h5) and necessary pickled files (onehot_encoded_geo.pkl, scalar.pkl, label_encoder_gender.pkl) in the same directory as app.py.
2. Open a terminal and navigate to the directory containing app.py.
3. Run the following command:
bash
 " streamlit run app.py " 
4. The application will open in your default web browser.

## User Input Fields
The application requires the following inputs to make predictions:
> Credit Score: Enter a numerical value.
> Geography: Select from predefined categories.
> Gender: Choose either Male or Female.
> Age: Use the slider to select an age between 18 and 48.
> Tenure: Use the slider to select a tenure between 0 and 10 years.
> Balance: Enter a numerical value for account balance.
> Number of Products: Select from 1 to 4 products.
> Has Credit Card: Choose between Yes (1) or No (0).
> Is Active Member: Choose between Yes (1) or No (0).
> Estimated Salary: Enter a numerical value.

## Prediction Output
After entering the required information, the application will display:
The probability of customer churn.
A message indicating whether the customer is likely to churn or not based on the prediction probability.

## Contributing
Feel free to fork this repository and submit pull requests for any improvements or features you would like to add!
